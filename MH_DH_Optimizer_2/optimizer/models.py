from pydantic import BaseModel, Field


class OptimizationConfig(BaseModel):
    max_hops: int = Field(default=4, ge=1, le=20)
    travel_time_threshold: float = Field(default=100_000, ge=0)
    max_allowed_combinations: int = Field(default=20_000_000, ge=1)
    service_time_minutes: int = Field(default=120, ge=0)
    recursion_limit: int = Field(default=5000, ge=1000, le=1_000_000)
    cost_40ft: float = Field(default=70, ge=0)
    cost_32ft: float = Field(default=50, ge=0)
    cost_24ft: float = Field(default=35, ge=0)
    cost_22ft: float = Field(default=35, ge=0)
    cost_20ft: float = Field(default=30, ge=0)
    cost_17ft: float = Field(default=30, ge=0)
    cost_14ft: float = Field(default=25, ge=0)
    cost_10ft: float = Field(default=25, ge=0)
    cost_8ft: float = Field(default=20, ge=0)
    cost_7ft: float = Field(default=20, ge=0)

    def vehicle_cost_per_km(self) -> dict[int, float]:
        return {
            40: self.cost_40ft,
            32: self.cost_32ft,
            24: self.cost_24ft,
            22: self.cost_22ft,
            20: self.cost_20ft,
            17: self.cost_17ft,
            14: self.cost_14ft,
            10: self.cost_10ft,
            8: self.cost_8ft,
            7: self.cost_7ft,
            0: 0.0,
        }
