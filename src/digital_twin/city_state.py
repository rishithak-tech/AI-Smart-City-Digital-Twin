from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CityState:
    """
    Represents the current state of the Smart City Digital Twin.
    """

    # ----------------------------------------
    # Location
    # ----------------------------------------

    city: str = "Hyderabad"


    # ----------------------------------------
    # Air Quality
    # ----------------------------------------

    current_aqi: Optional[float] = None
    predicted_aqi_next_hour: Optional[float] = None


    # ----------------------------------------
    # Weather
    # ----------------------------------------

    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    surface_pressure: Optional[float] = None


    # ----------------------------------------
    # Traffic
    # ----------------------------------------

    traffic_density_score: Optional[float] = None
    vehicle_pixel_ratio: Optional[float] = None


    # ----------------------------------------
    # Risk Indicators
    # ----------------------------------------

    air_quality_risk: Optional[str] = None
    traffic_risk: Optional[str] = None


    # ----------------------------------------
    # Convert to dictionary
    # ----------------------------------------

    def to_dict(self):

        return asdict(self)