import os
from functools import lru_cache
from typing import Dict, Optional
import polars as pl
_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "airports.csv")
@lru_cache(maxsize=1)
def _airport_lookup() -> Dict[str, Dict[str, Optional[float]]]:
    """
    Load airport coordinates indexed by IATA code.
    """
    if not os.path.exists(_DATA_PATH):
        return {}
    try:
        df = pl.read_csv(
            _DATA_PATH,
            infer_schema_length=1000,
            columns=["iata_code", "latitude_deg", "longitude_deg"],
        )
    except Exception:
        return {}
    if df.is_empty():
        return {}
    df = (
        df.drop_nulls("iata_code")
        .with_columns(pl.col("iata_code").str.to_uppercase())
        .filter(pl.col("iata_code").str.len_chars() > 0)
        .unique(subset=["iata_code"], keep="first")
    )
    lookup: Dict[str, Dict[str, Optional[float]]] = {}
    for row in df.iter_rows(named=True):
        code = row["iata_code"]
        try:
            lat = float(row["latitude_deg"]) if row["latitude_deg"] is not None else None
        except (TypeError, ValueError):
            lat = None
        try:
            lon = float(row["longitude_deg"]) if row["longitude_deg"] is not None else None
        except (TypeError, ValueError):
            lon = None
        lookup[code] = {
            "airport_code": code,
            "latitude": lat,
            "longitude": lon,
        }
    return lookup
def get_airport_coordinates(iata_code: Optional[str]) -> Dict[str, Optional[float]]:
    """
    Return latitude/longitude for an IATA code. Unknown codes return None coordinates.
    """
    if not iata_code:
        return {
            "airport_code": iata_code,
            "latitude": None,
            "longitude": None,
        }
    code = iata_code.upper()
    lookup = _airport_lookup()
    return lookup.get(
        code,
        {
            "airport_code": code,
            "latitude": None,
            "longitude": None,
        },
    )