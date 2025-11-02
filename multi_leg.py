from __future__ import annotations
import polars as pl
from datetime import datetime, timedelta
from math import inf
from typing import Optional, List, Dict, Tuple, Iterable
import time
import weakref

def _parse_utc_datetime(column: str) -> pl.Expr:
    """Parse a UTC timestamp column into a Polars Datetime (naive UTC)."""
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f", strict=False)
    )

def with_connection_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """
    Input columns (strings/ints as specified):
      - SCHEDULED_DEPARTURE_DATE_TIME_UTC (str/datetime)
      - SCHEDULED_ARRIVAL_DATE_TIME_UTC (str/datetime)
      - Additional schedule metadata is preserved untouched.
    Output:
      - Adds dep_ts (Datetime), arr_ts (Datetime), keeping originals.
    """
    return df.with_columns(
        [
            _parse_utc_datetime("SCHEDULED_DEPARTURE_DATE_TIME_UTC").alias("dep_ts"),
            _parse_utc_datetime("SCHEDULED_ARRIVAL_DATE_TIME_UTC").alias("arr_ts"),
        ]
    )

# ---------- Connection Scan Algorithm (CSA) ----------

class Connection(Tuple):
    __slots__ = ()
    # Structure:
    # (dep_ts, arr_ts, dep_airport, arr_airport, carrier, flight_number, row_idx)

_build_connections_cache: Dict[int, Tuple[weakref.ref, List[Tuple[datetime, datetime, str, str, Optional[str], Optional[str], int]]]] = {}

def _try_get_cached_connections(df: pl.DataFrame):
    key = id(df)
    entry = _build_connections_cache.get(key)
    if not entry:
        return None
    df_ref, conns = entry
    cached_df = df_ref()
    if cached_df is None or cached_df is not df:
        # Underlying DataFrame is gone or replaced; drop the stale cache entry.
        _build_connections_cache.pop(key, None)
        return None
    return conns

def build_connections(df: pl.DataFrame) -> List[Tuple[datetime, datetime, str, str, Optional[str], Optional[str], int]]:
    """
    Convert to a list of connections sorted by departure time.
    Each connection is a tuple: (dep_ts, arr_ts, dep_airport, arr_airport, carrier, flight_number, row_idx)
    """
    cached = _try_get_cached_connections(df)
    if cached is not None:
        return cached

    # Select only the needed columns to minimize Python overhead and sort in Polars
    slim = (
        df.select(
            "dep_ts", "arr_ts", "DEPAPT", "ARRAPT", "CARRIER", "FLTNO"
        )
        .with_row_index(name="__row_idx__")
        .sort("dep_ts")
    )

    if slim.height:
        # Bring to Python lists (fast; avoids per-row .to_dict cost)
        dep_ts = slim["dep_ts"].to_list()
        arr_ts = slim["arr_ts"].to_list()
        dep_ap = slim["DEPAPT"].to_list()
        arr_ap = slim["ARRAPT"].to_list()
        carriers = slim["CARRIER"].to_list()
        flight_numbers = slim["FLTNO"].to_list()
        idxs = slim["__row_idx__"].to_list()

        conns = list(zip(dep_ts, arr_ts, dep_ap, arr_ap, carriers, flight_numbers, idxs))
    else:
        conns = []
    _build_connections_cache[id(df)] = (weakref.ref(df), conns)
    return conns

def itinerary_travel_time(itinerary: List[Dict]) -> Optional[timedelta]:
    """
    Compute total journey time for an itinerary using the first departure and final arrival.
    """
    if not itinerary:
        return None
    first_departure = itinerary[0]["dep_ts"]
    final_arrival = itinerary[-1]["arr_ts"]
    return final_arrival - first_departure

# df must already have dep_ts, arr_ts (use your with_connection_timestamps once)

def fastest_leq_one_stop(
    df: pl.DataFrame,
    origin: str,
    destination: str,
    earliest_departure: datetime,
    latest_arrival: datetime,
    min_connection: timedelta = timedelta(minutes=30),
    min_origin_buffer: timedelta = timedelta(0),
    streaming: bool = True,
):
    # Keep only needed columns and add stable row index
    cols = [
        "DEPAPT","ARRAPT","CARRIER","FLTNO",
        "FLIGHT_DATE","DEPTIM","ARRTIM","ARRDAY",
        "dep_ts","arr_ts",
    ]
    base = (
        df.select(cols)
          .with_row_index("__idx")
          .with_columns([
              pl.col("DEPAPT","ARRAPT","CARRIER","FLTNO").cast(pl.Categorical)
          ])
          .lazy()
    )

    origin_ready = earliest_departure + min_origin_buffer

    # Direct option
    direct = (
        base
        .filter(
            (pl.col("DEPAPT") == origin) &
            (pl.col("ARRAPT") == destination) &
            (pl.col("dep_ts") >= origin_ready) &
            (pl.col("arr_ts") <= latest_arrival)
        )
        .with_columns([
            (pl.col("arr_ts") - pl.col("dep_ts")).alias("total_time"),
            pl.lit(0).alias("stops"),
            pl.lit(None, dtype=pl.Categorical).alias("via"),
            pl.col("__idx").alias("idx1"),
            pl.lit(None, dtype=pl.Int64).alias("idx2"),
            pl.col("DEPAPT").alias("o"),
            pl.col("ARRAPT").alias("d"),
            pl.col("dep_ts").alias("dep_ts_out"),
            pl.col("arr_ts").alias("arr_ts_in"),
        ])
        .select([
            "stops","via","o","d","dep_ts_out","arr_ts_in","total_time",
            "CARRIER","FLTNO","FLIGHT_DATE","DEPTIM","ARRTIM","ARRDAY",
            "idx1","idx2"
        ])
    )

    # One-stop: origin leg (A→K)
    f1 = (
        base
        .filter(
            (pl.col("DEPAPT") == origin) &
            (pl.col("dep_ts") >= origin_ready) &
            # must still allow a connection and arrive by cutoff
            (pl.col("arr_ts") <= latest_arrival - pl.lit(min_connection))
        )
        .select([
            pl.col("__idx").alias("idx1"),
            pl.col("DEPAPT").alias("o1"),
            pl.col("ARRAPT").alias("k"),
            pl.col("CARRIER").alias("car1"),
            pl.col("FLTNO").alias("flt1"),
            "FLIGHT_DATE","DEPTIM","ARRTIM","ARRDAY",
            pl.col("dep_ts").alias("dep1"),
            pl.col("arr_ts").alias("arr1"),
        ])
    )

    # One-stop: inbound leg (K→B)
    f2 = (
        base
        .filter(
            (pl.col("ARRAPT") == destination) &
            (pl.col("arr_ts") <= latest_arrival)
        )
        .select([
            pl.col("__idx").alias("idx2"),
            pl.col("DEPAPT").alias("k2"),
            pl.col("ARRAPT").alias("d2"),
            pl.col("CARRIER").alias("car2"),
            pl.col("FLTNO").alias("flt2"),
            pl.col("dep_ts").alias("dep2"),
            pl.col("arr_ts").alias("arr2"),
        ])
    )

    one_stop = (
        f1.join(f2, left_on="k", right_on="k2", how="inner")
          .filter(pl.col("dep2") >= pl.col("arr1") + pl.lit(min_connection))
          .with_columns([
              (pl.col("arr2") - pl.col("dep1")).alias("total_time"),
              pl.lit(1).alias("stops"),
              pl.col("k").alias("via"),
              pl.col("o1").alias("o"),
              pl.col("d2").alias("d"),
              pl.col("dep1").alias("dep_ts_out"),
              pl.col("arr2").alias("arr_ts_in"),
          ])
          .select([
              "stops","via","o","d","dep_ts_out","arr_ts_in","total_time",
              # take carrier/flight numbers for both legs; keep names distinct
              "car1","flt1","car2","flt2",
              # optional metadata from first leg; add more if needed
              "FLIGHT_DATE","DEPTIM","ARRTIM","ARRDAY",
              "idx1","idx2"
          ])
    )

    best = (
        pl.concat([direct, one_stop], how="diagonal_relaxed")
          .sort(["total_time","arr_ts_in"])
          .limit(1)
          .collect(streaming=streaming)
    )

    if best.height == 0:
        return None, None

    row = best.row(0, named=True)
    if row["stops"] == 0:
        # single leg
        leg = df.row(int(row["idx1"]), named=True)
        itinerary = [{
            "row_idx": int(row["idx1"]),
            "DEPAPT": row["o"],
            "ARRAPT": row["d"],
            "CARRIER": leg.get("CARRIER"),
            "FLTNO": leg.get("FLTNO"),
            "dep_ts": row["dep_ts_out"],
            "arr_ts": row["arr_ts_in"],
            "FLIGHT_DATE": leg.get("FLIGHT_DATE"),
            "DEPTIM": leg.get("DEPTIM"),
            "ARRTIM": leg.get("ARRTIM"),
            "ARRDAY": leg.get("ARRDAY"),
            "ESTIMATED_CO2_PER_CAPITA": leg.get("ESTIMATED_CO2_PER_CAPITA"),
            "ESTIMATED_CO2_TOTAL_TONNES": leg.get("ESTIMATED_CO2_TOTAL_TONNES"),
            "SEATS": leg.get("SEATS"),
            "TOTAL_SEATS": leg.get("TOTAL_SEATS"),
        }]
    else:
        leg1 = df.row(int(row["idx1"]), named=True)
        leg2 = df.row(int(row["idx2"]), named=True)
        itinerary = [
            {
                "row_idx": int(row["idx1"]),
                "DEPAPT": row["o"],
                "ARRAPT": row["via"],
                "CARRIER": leg1.get("CARRIER"),
                "FLTNO": leg1.get("FLTNO"),
                "dep_ts": leg1.get("dep_ts"),
                "arr_ts": leg1.get("arr_ts"),
                "FLIGHT_DATE": leg1.get("FLIGHT_DATE"),
                "DEPTIM": leg1.get("DEPTIM"),
                "ARRTIM": leg1.get("ARRTIM"),
                "ARRDAY": leg1.get("ARRDAY"),
                "ESTIMATED_CO2_PER_CAPITA": leg1.get("ESTIMATED_CO2_PER_CAPITA"),
                "ESTIMATED_CO2_TOTAL_TONNES": leg1.get("ESTIMATED_CO2_TOTAL_TONNES"),
                "SEATS": leg1.get("SEATS"),
                "TOTAL_SEATS": leg1.get("TOTAL_SEATS"),
            },
            {
                "row_idx": int(row["idx2"]),
                "DEPAPT": row["via"],
                "ARRAPT": row["d"],
                "CARRIER": leg2.get("CARRIER"),
                "FLTNO": leg2.get("FLTNO"),
                "dep_ts": leg2.get("dep_ts"),
                "arr_ts": leg2.get("arr_ts"),
                "FLIGHT_DATE": leg2.get("FLIGHT_DATE"),
                "DEPTIM": leg2.get("DEPTIM"),
                "ARRTIM": leg2.get("ARRTIM"),
                "ARRDAY": leg2.get("ARRDAY"),
                "ESTIMATED_CO2_PER_CAPITA": leg2.get("ESTIMATED_CO2_PER_CAPITA"),
                "ESTIMATED_CO2_TOTAL_TONNES": leg2.get("ESTIMATED_CO2_TOTAL_TONNES"),
                "SEATS": leg2.get("SEATS"),
                "TOTAL_SEATS": leg2.get("TOTAL_SEATS"),
            },
        ]

    return itinerary, row["arr_ts_in"]
