from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import OptimizationConfig

REQUIRED_LOCATION_COLS = [
    "location_name",
    "demand",
    "ML",
    "Freq_Allowed",
    "depot_departure (minutes)",
    "time_window_end (minutes)",
    "latitude",
    "longitude",
]
REQUIRED_DISTANCE_COLS = ["location 1", "location 2", "distance (km)"]
REQUIRED_TIME_COLS = ["location 1", "location 2", "travel_time (minutes)"]

DAY_MINUTES = 24 * 60


def _minutes_to_clock_hhmm(minutes: float) -> str:
    """24h clock time HH:MM from a minute offset.

    Totals of 24+ hours are folded to time-of-day only (minutes within one day),
    e.g. 1800 minutes -> 06:00, not 30:00.
    """
    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)):
        return ""
    total = int(round(float(minutes)))
    within_day = total % DAY_MINUTES
    h, mn = divmod(within_day, 60)
    return f"{h:02d}:{mn:02d}"


def _format_final_assignment_roundup(df: pd.DataFrame) -> pd.DataFrame:
    """dist, monthly_cost, total_demand as integers (rounded)."""
    if df.empty:
        return df
    out = df.copy()
    lower = {str(c).lower(): c for c in out.columns}

    def col(name: str) -> str | None:
        return lower.get(name.lower())

    for logical in ("dist", "monthly_cost", "total_demand"):
        c = col(logical)
        if not c:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = s.apply(lambda v: pd.NA if pd.isna(v) else int(round(float(v))))
        out[c] = out[c].astype("Int64")
    return out


def _format_final_assignment_time_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert arrival_times / departure_times (colon-separated minutes) and updated_depot_departure to HH:MM."""
    if df.empty:
        return df
    out = df.copy()
    lower = {str(c).lower(): c for c in out.columns}

    def fold_colon_times(cell: object) -> object:
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return cell
        s = str(cell).strip()
        if not s:
            return cell
        parts = [p.strip() for p in s.split(":") if p.strip() != ""]
        if not parts:
            return cell
        converted = []
        for p in parts:
            try:
                converted.append(_minutes_to_clock_hhmm(float(p)))
            except ValueError:
                converted.append(p)
        return " | ".join(converted)

    for logical in ("arrival_times", "departure_times"):
        key = lower.get(logical.lower())
        if key and key in out.columns:
            out[key] = out[key].apply(fold_colon_times)

    udd = lower.get("updated_depot_departure")
    if udd and udd in out.columns:
        out[udd] = pd.to_numeric(out[udd], errors="coerce").apply(
            lambda v: "" if pd.isna(v) else _minutes_to_clock_hhmm(float(v)),
        )
    return out


def _format_expanded_schedule_display(df: pd.DataFrame) -> pd.DataFrame:
    """Arrival / departure as HH:MM (time of day); Total_Demand as integer."""
    if df.empty:
        return df
    out = df.copy()
    lower = {str(c).lower(): c for c in out.columns}

    def fmt_time_cell(v: object) -> str:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return ""
        if pd.isna(v):
            return ""
        try:
            return _minutes_to_clock_hhmm(float(v))
        except (TypeError, ValueError):
            return str(v)

    for logical in ("arrival_time", "departure_time"):
        key = lower.get(logical)
        if not key:
            continue
        series = out[key].map(fmt_time_cell)
        out[key] = series.astype("string")

    td = lower.get("total_demand")
    if td:
        s = pd.to_numeric(out[td], errors="coerce")
        out[td] = s.apply(lambda v: pd.NA if pd.isna(v) else int(round(float(v))))
        out[td] = out[td].astype("Int64")

    return out


def validate_input_frames(
    location_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    time_df: pd.DataFrame,
) -> None:
    for col in REQUIRED_LOCATION_COLS:
        if col not in location_df.columns:
            raise ValueError(f"Location_file missing required column: {col!r}")
    for col in REQUIRED_DISTANCE_COLS:
        if col not in distance_df.columns:
            raise ValueError(f"Distance_Matrix missing required column: {col!r}")
    for col in REQUIRED_TIME_COLS:
        if col not in time_df.columns:
            raise ValueError(f"Time_Matrix missing required column: {col!r}")
    if len(location_df) < 2:
        raise ValueError("Location_file must have at least a depot row and one destination.")


@dataclass
class OptimizationResult:
    clustering_output: pd.DataFrame
    filtered_routes: pd.DataFrame
    final_assignment: pd.DataFrame
    expanded_schedule: pd.DataFrame
    logs: list[str] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None


def calculate_bearing(lat1, lon1, lat2, lon2):
    delta_lon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def assign_vehicle_length(total_demand):
    if total_demand > 2200:
        return 40
    if total_demand > 1550:
        return 32
    if total_demand > 1325:
        return 24
    if total_demand > 1255:
        return 22
    if total_demand > 893:
        return 20
    if total_demand > 686:
        return 17
    if total_demand > 400:
        return 14
    if total_demand > 250:
        return 10
    if total_demand > 180:
        return 8
    if total_demand > 0:
        return 7
    return 0


def find_best_combination(remaining_hubs, available_routes, memo):
    if not remaining_hubs:
        return 0, []
    state = tuple(sorted(list(remaining_hubs)))
    if state in memo:
        return memo[state]

    best_cost = float("inf")
    best_set = []
    target_hub = next(iter(remaining_hubs))

    potential_routes = [r for r in available_routes if target_hub in r["hubs_set"]]
    potential_routes.sort(key=lambda x: x["monthly_cost"])

    for route in potential_routes:
        if route["hubs_set"].issubset(remaining_hubs):
            new_hubs = remaining_hubs - route["hubs_set"]
            cost_of_rest, set_of_rest = find_best_combination(new_hubs, available_routes, memo)
            if cost_of_rest != float("inf"):
                total_cost = route["monthly_cost"] + cost_of_rest
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_set = [route] + set_of_rest
    memo[state] = (best_cost, best_set)
    return best_cost, best_set


def _log(logs: list[str], msg: str) -> None:
    logs.append(msg)


def run_optimization(
    location_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    time_df: pd.DataFrame,
    config: OptimizationConfig,
) -> OptimizationResult:
    logs: list[str] = []
    validate_input_frames(location_df, distance_df, time_df)

    locs = location_df.copy().dropna(how="all")
    dist_df = distance_df.copy().dropna(how="all")
    time_df_clean = time_df.copy().dropna(how="all")

    max_hops = config.max_hops
    max_allowed_combinations = config.max_allowed_combinations
    vehicle_cost = config.vehicle_cost_per_km()
    service_minutes = float(config.service_time_minutes)

    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(config.recursion_limit)

        _log(logs, "Step 1: Dynamic Clustering...")
        numeric_cols = [
            "demand",
            "ML",
            "Freq_Allowed",
            "depot_departure (minutes)",
            "time_window_end (minutes)",
            "latitude",
            "longitude",
        ]
        for col in numeric_cols:
            if col in locs.columns:
                locs[col] = pd.to_numeric(locs[col], errors="coerce")
                if col in ["demand", "ML", "Freq_Allowed"]:
                    locs[col] = locs[col].fillna(0)
                elif col in ["latitude", "longitude"]:
                    locs[col] = locs[col].fillna(0)

        depot = locs.iloc[0]
        destinations = locs.iloc[1:].copy()
        destinations["bearing"] = destinations.apply(
            lambda r: calculate_bearing(depot["latitude"], depot["longitude"], r["latitude"], r["longitude"]),
            axis=1,
        )
        destinations = destinations.sort_values("bearing").reset_index(drop=True)
        n = len(destinations)
        for k in range(1, n + 1):
            group_ids = np.array_split(np.arange(n), k)
            labels = np.zeros(n, dtype=int)
            for gid, idxs in enumerate(group_ids):
                labels[idxs] = gid
            total_comb = sum(math.perm(len(idxs), min(len(idxs), max_hops)) for idxs in group_ids)
            if total_comb <= max_allowed_combinations:
                destinations["bearing_group"] = labels
                break
        destinations["final_group"] = (
            destinations["depot_departure (minutes)"].astype(str) + "-" + destinations["bearing_group"].astype(str)
        )
        clustered_df = pd.concat([pd.DataFrame([depot]), destinations], ignore_index=True)

        _log(logs, "Step 2: Generating Routes & Pruning Identical Sets...")
        dist_dict = {
            (str(r["location 1"]), str(r["location 2"])): float(r["distance (km)"])
            for _, r in dist_df.iterrows()
        }
        attr = clustered_df.set_index("location_name").to_dict("index")
        depot_name = str(clustered_df.iloc[0]["location_name"])

        raw_routes = []
        groups = destinations["final_group"].unique()
        for gid in groups:
            grp_hubs = destinations[destinations["final_group"] == gid]["location_name"].astype(str).tolist()
            for h in range(1, max_hops + 1):
                for perm in itertools.permutations(grp_hubs, h):
                    route_seq = [depot_name] + list(perm) + [depot_name]
                    d_total = 0.0
                    path_ok = True
                    for i in range(len(route_seq) - 1):
                        d = dist_dict.get((route_seq[i], route_seq[i + 1]))
                        if d is None:
                            path_ok = False
                            break
                        d_total += d
                    if path_ok:
                        raw_routes.append(
                            {
                                "route_sequence": " -> ".join(route_seq),
                                "hubs": list(perm),
                                "hubs_set": set(perm),
                                "dist": d_total,
                                "group": gid,
                            }
                        )

        _log(logs, "Step 3: Calculating Absolute Costs & Domination Pruning...")
        costed_map = {}
        for r in raw_routes:
            base_demand = sum(attr[h]["demand"] for h in r["hubs"])
            max_ml = min(attr[h]["ML"] for h in r["hubs"])
            freq_ok = all(attr[h].get("Freq_Allowed", 0) == 1 for h in r["hubs"])

            v1 = assign_vehicle_length(base_demand)
            c1 = (
                (r["dist"] * vehicle_cost.get(v1, 999)) * 30
                if v1 <= max_ml
                else float("inf")
            )
            v2 = assign_vehicle_length(base_demand * 2)
            c2 = (
                (r["dist"] * vehicle_cost.get(v2, 999)) * 15
                if freq_ok and v2 <= max_ml
                else float("inf")
            )

            if c1 == float("inf") and c2 == float("inf"):
                continue

            m_cost = min(c1, (1.1 * c2))
            f_val = 2 if (1.1 * c2) < c1 else 1
            t_dem = base_demand * f_val
            v_len = assign_vehicle_length(t_dem)

            h_key = tuple(sorted(list(r["hubs_set"])))
            if h_key not in costed_map or m_cost < costed_map[h_key]["monthly_cost"]:
                r.update(
                    {
                        "monthly_cost": m_cost,
                        "Freq": f_val,
                        "total_demand": t_dem,
                        "assigned_vehicle_length": v_len,
                    }
                )
                costed_map[h_key] = r

        costed_routes = list(costed_map.values())

        filtered_rows = []
        for r in costed_routes:
            row = dict(r)
            hs = row.pop("hubs_set", None)
            if hs is not None:
                row["hubs_set"] = ",".join(sorted(hs))
            filtered_rows.append(row)
        filtered_df = pd.DataFrame(filtered_rows)

        _log(logs, "Step 4: Global Minimum Cost Selection (Set Cover)...")
        final_assigned = []
        for gid in groups:
            grp_routes = [r for r in costed_routes if r["group"] == gid]
            grp_hubs = set(
                destinations[destinations["final_group"] == gid]["location_name"].astype(str).tolist()
            )

            _log(logs, f"Processing Group {gid}: {len(grp_hubs)} hubs, {len(grp_routes)} potential routes...")

            _, best_set = find_best_combination(grp_hubs, grp_routes, {})

            if not best_set and grp_hubs:
                _log(logs, f"CRITICAL ERROR: No valid combination found for Group {gid}!")
                all_covered_hubs = set().union(*(r["hubs_set"] for r in grp_routes))
                missing = grp_hubs - all_covered_hubs
                if missing:
                    _log(logs, f"Reason: The following hubs have NO valid routes: {missing}")
            else:
                _log(logs, f"Success: Group {gid} optimized with {len(best_set)} routes.")
                final_assigned.extend(best_set)

        if not final_assigned:
            _log(logs, "CRITICAL: final_assigned is completely empty.")
            return OptimizationResult(
                clustering_output=clustered_df,
                filtered_routes=filtered_df,
                final_assignment=pd.DataFrame(),
                expanded_schedule=pd.DataFrame(),
                logs=logs,
                success=False,
                error_message="No routes could be assigned. Check distance matrix coverage and constraints.",
            )

        _log(logs, "Step 6: Time Shifting & Metadata Restoration...")
        time_dict = {
            (str(r["location 1"]), str(r["location 2"])): float(r["travel_time (minutes)"])
            for _, r in time_df_clean.iterrows()
        }

        expanded_rows = []
        final_assignment_rows = []

        for idx, row in enumerate(final_assigned):
            stops = [s.strip() for s in row["route_sequence"].split("->")]
            base_dep = float(attr[stops[1]]["depot_departure (minutes)"])

            temp_t = base_dep
            r_times = []
            for i in range(len(stops) - 1):
                arr = temp_t + time_dict.get((stops[i], stops[i + 1]), 0.0)
                if stops[i + 1] != depot_name:
                    r_times.append({"DH": stops[i + 1], "arr": arr, "dep": arr + service_minutes})
                    temp_t = arr + service_minutes
                else:
                    r_times.append({"DH": stops[i + 1], "arr": arr, "dep": np.nan})

            buffers = [
                float(attr[t["DH"]]["time_window_end (minutes)"]) - t["arr"]
                for t in r_times
                if t["DH"] != depot_name
            ]
            shift = max(0, min(buffers)) if buffers else 0

            arr_str = ":".join([str(round(t["arr"] + shift, 2)) for t in r_times])
            dep_str = ":".join([str(round(t["dep"] + shift, 2)) for t in r_times if not np.isnan(t["dep"])])

            row_summary = {k: v for k, v in row.items() if k != "hubs_set"}
            row_summary.update(
                {
                    "Route_ID": idx + 1,
                    "arrival_times": arr_str,
                    "departure_times": dep_str,
                    "updated_depot_departure": round(base_dep + shift, 2),
                }
            )
            final_assignment_rows.append(row_summary)

            for i, stop in enumerate(stops):
                expanded_rows.append(
                    {
                        "Route_ID": idx + 1,
                        "Location": stop,
                        "Arrival_Time": np.nan
                        if i == 0
                        else round(r_times[i - 1]["arr"] + shift, 2),
                        "Departure_Time": round(base_dep + shift, 2)
                        if i == 0
                        else (
                            round(r_times[i - 1]["dep"] + shift, 2)
                            if i < len(stops) - 1
                            else np.nan
                        ),
                        "Freq": row["Freq"],
                        "Vehicle_Length": row["assigned_vehicle_length"],
                        "Total_Demand": row["total_demand"],
                        "Route_Sequence": row["route_sequence"],
                    }
                )

        final_assignment_df = _format_final_assignment_roundup(pd.DataFrame(final_assignment_rows))
        final_assignment_df = _format_final_assignment_time_strings(final_assignment_df)
        expanded_df = _format_expanded_schedule_display(pd.DataFrame(expanded_rows))

        _log(logs, "Pipeline complete.")
        return OptimizationResult(
            clustering_output=clustered_df,
            filtered_routes=filtered_df,
            final_assignment=final_assignment_df,
            expanded_schedule=expanded_df,
            logs=logs,
            success=True,
            error_message=None,
        )
    finally:
        sys.setrecursionlimit(old_limit)
