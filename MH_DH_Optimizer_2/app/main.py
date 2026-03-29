from __future__ import annotations

import json
import os
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from time import time

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.templating import Jinja2Templates

from optimizer.models import OptimizationConfig
from optimizer.pipeline import run_optimization

BASE_DIR = Path(__file__).resolve().parent

REQUIRED_SHEETS = ("Location_file", "Distance_Matrix", "Time_Matrix")
SESSION_TTL_SEC = 3600
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_sessions: dict[str, dict] = {}


def _cleanup_sessions() -> None:
    now = time()
    dead = [sid for sid, meta in _sessions.items() if now - meta["created"] > SESSION_TTL_SEC]
    for sid in dead:
        p = _sessions.pop(sid, None)
        if p and Path(p["path"]).exists():
            try:
                Path(p["path"]).unlink()
            except OSError:
                pass


def _read_input_workbook(content: bytes) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        sheets = pd.read_excel(
            BytesIO(content),
            sheet_name=list(REQUIRED_SHEETS),
            engine="openpyxl",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read workbook. Ensure sheets exist: {', '.join(REQUIRED_SHEETS)}. ({e})",
        ) from e
    loc = _normalize_location_sheet(sheets["Location_file"].dropna(how="all"))
    dist = sheets["Distance_Matrix"].dropna(how="all")
    tim = sheets["Time_Matrix"].dropna(how="all")
    return loc, dist, tim


def _normalize_location_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Strip headers and map common aliases so latitude/longitude/location_name are found."""
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lower_to_col = {c.lower(): c for c in out.columns}
    aliases: list[tuple[str, tuple[str, ...]]] = [
        (
            "location_name",
            ("location_name", "location", "location name", "facility", "hub", "name"),
        ),
        ("latitude", ("latitude", "lat")),
        ("longitude", ("longitude", "long", "lng", "lon")),
    ]
    rename: dict[str, str] = {}
    for standard, candidates in aliases:
        if standard in out.columns:
            continue
        for cand in candidates:
            col = lower_to_col.get(cand.lower())
            if col is not None:
                rename[col] = standard
                break
    if rename:
        out = out.rename(columns=rename)
    return out


def _df_to_json_records(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _build_workbook_bytes(
    location_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    time_df: pd.DataFrame,
    clustering: pd.DataFrame,
    filtered: pd.DataFrame,
    final_assignment: pd.DataFrame,
    expanded: pd.DataFrame,
) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        location_df.to_excel(w, sheet_name="Location_file", index=False)
        distance_df.to_excel(w, sheet_name="Distance_Matrix", index=False)
        time_df.to_excel(w, sheet_name="Time_Matrix", index=False)
        clustering.to_excel(w, sheet_name="Clustering_Output", index=False)
        filtered.to_excel(w, sheet_name="Filtered_Routes", index=False)
        final_assignment.to_excel(w, sheet_name="Final_Assignment", index=False)
        expanded.to_excel(w, sheet_name="Expanded_Schedule", index=False)
    buf.seek(0)
    return buf.getvalue()


def _run_job(
    location_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    time_df: pd.DataFrame,
    config: OptimizationConfig,
):
    return run_optimization(location_df, distance_df, time_df, config)


app = FastAPI(title="MH-DH Optimizer", version="1.0.0")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

static_dir = BASE_DIR / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/api/optimize")
async def api_optimize(
    file: UploadFile = File(...),
    config: str = Form(...),
):
    _cleanup_sessions()

    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload an .xlsx file.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB).")

    try:
        payload = json.loads(config)
        if not isinstance(payload, dict):
            raise TypeError("Config must be a JSON object")
        _static_limits = OptimizationConfig()
        merged = {**_static_limits.model_dump(), **payload}
        merged["max_allowed_combinations"] = _static_limits.max_allowed_combinations
        merged["recursion_limit"] = _static_limits.recursion_limit
        opt_config = OptimizationConfig.model_validate(merged)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration JSON: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}") from e

    try:
        location_df, distance_df, time_df = _read_input_workbook(raw)
    except HTTPException:
        raise

    try:
        result = await run_in_threadpool(
            _run_job,
            location_df,
            distance_df,
            time_df,
            opt_config,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e!s}") from e

    xlsx_bytes = _build_workbook_bytes(
        location_df,
        distance_df,
        time_df,
        result.clustering_output,
        result.filtered_routes,
        result.final_assignment,
        result.expanded_schedule,
    )

    session_id = uuid.uuid4().hex
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.write(xlsx_bytes)
    tmp.close()
    _sessions[session_id] = {"path": tmp.name, "created": time()}

    return JSONResponse(
        {
            "success": result.success,
            "session_id": session_id,
            "logs": result.logs,
            "error_message": result.error_message,
            "expanded_schedule": _df_to_json_records(result.expanded_schedule),
            "final_assignment": _df_to_json_records(result.final_assignment),
        }
    )


@app.get("/api/download/{session_id}")
async def api_download(session_id: str):
    _cleanup_sessions()
    meta = _sessions.get(session_id)
    if not meta or not Path(meta["path"]).exists():
        raise HTTPException(status_code=404, detail="Report expired or not found.")
    return FileResponse(
        meta["path"],
        filename="MH_DH_Optimizer_report.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=port,
        reload=True,
    )
