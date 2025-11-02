from __future__ import annotations
import polars as pl
from datetime import datetime, timedelta
from math import inf
from typing import Optional, List, Dict, Tuple, Iterable
import time
import weakref

# ---------- Timestamp helpers for OAG-like HHMM + ARRDAY ----------

def _hhmm_duration_expr(column: str) -> pl.Expr:
    """Return a duration expression representing HHMM values stored as integers."""
    base = pl.col(column).cast(pl.Int64)
    return pl.duration(hours=(base // 100), minutes=(base % 100))

def with_connection_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """
    Input columns (strings/ints as specified):
      - DEPAPT (str), ARRAPT (str)
      - FLIGHT_DATE (YYYY-MM-DD str)
      - DEPTIM (int HHMM)
      - ARRTIM (int HHMM)
      - ARRDAY (str marker: 'P', '1', '2', ... or empty)
      - CARRIER (str), FLTNO (str/int)
    Output:
      - Adds dep_ts (Datetime), arr_ts (Datetime), keeps originals.
    """
    out = (
        df
        .with_columns(
            # Parse date
            pl.col("FLIGHT_DATE").str.strptime(pl.Date, "%Y-%m-%d").alias("dep_date"),
            # Normalize ARRDAY to integer day offset
            pl.col("ARRDAY")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars()
            .alias("ARRDAY_norm"),
        )
        .with_columns(
            # Departure timestamp = dep_date + HHMM
            (pl.col("dep_date").cast(pl.Datetime) + _hhmm_duration_expr("DEPTIM")).alias("dep_ts"),
            # Arrival timestamp = dep_date + day offset + HHMM
            (
                pl.col("dep_date").cast(pl.Datetime)
                + pl.duration(
                    days=(
                        pl.when(pl.col("ARRDAY_norm") == "P")
                        .then(pl.lit(-1))
                        .otherwise(
                            pl.col("ARRDAY_norm").cast(pl.Int32, strict=False)
                        )
                        .fill_null(0)
                    )
                )
                + _hhmm_duration_expr("ARRTIM")
            ).alias("arr_ts"),
        )
        .drop(["dep_date", "ARRDAY_norm"])
    )
    return out

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


def csa_fastest_route(
    df: pl.DataFrame,
    origin: str,
    destination: str,
    earliest_departure: datetime,
    latest_arrival: datetime,
    min_connection: timedelta = timedelta(minutes=30),
    min_origin_buffer: timedelta = timedelta(0),
    focus_airports: Optional[Iterable[str]] = None,
) -> Tuple[Optional[List[Dict]], Optional[datetime]]:
    """
    Single-pass Connection Scan Algorithm:
      - Scans connections in increasing departure time
      - Respects a minimum connection time (no MCT at origin unless you set min_origin_buffer)
      - Returns (itinerary, arrival_time) or (None, None) if not reachable by latest_arrival

    Returns `itinerary` as list of dicts with the original row index and key fields.
    """
    # Pre-filter to a time window to keep the scan lean:
    # Keep connections that depart no earlier than (earliest_departure - min_origin_buffer)
    # and arrive no later than latest_arrival.
    t_start = time.perf_counter()
    window = df.filter(
        (pl.col("arr_ts") <= pl.lit(latest_arrival))
        & (pl.col("dep_ts") >= pl.lit(earliest_departure - min_origin_buffer))
    )
    if focus_airports:
        focus_list = list(dict.fromkeys(focus_airports))
        window = window.filter(
            pl.col("DEPAPT").is_in(focus_list)
            | pl.col("ARRAPT").is_in(focus_list)
        )
    print(
        f"[CSA] Filter window retained {window.height} rows "
        f"in {time.perf_counter() - t_start:.4f}s"
    )

    if window.height == 0:
        return None, None

    # Build scan list
    t_build = time.perf_counter()
    conns = build_connections(window)
    print(
        f"[CSA] Built {len(conns)} connections "
        f"in {time.perf_counter() - t_build:.4f}s"
    )

    # Label sets: earliest known time you can be at airport a
    earliest: Dict[str, datetime] = {}
    # Available-from time accounting for connection slack per stop
    available_from: Dict[str, datetime] = {}

    # Initialize
    earliest[origin] = earliest_departure
    # You can board the first flight out of origin any time >= earliest_departure + min_origin_buffer
    available_from[origin] = earliest_departure + min_origin_buffer

    # Predecessors for path reconstruction: for each airport, remember the connection that improved it
    # pred[airport] = (prev_airport, connection_tuple)
    pred: Dict[str, Tuple[str, Tuple[datetime, datetime, str, str, Optional[str], Optional[str], int]]] = {}

    # Scan
    t_scan = time.perf_counter()
    relaxations = 0
    for dep_ts, arr_ts, u, v, carrier, flight_number, row_idx in conns:
        # Skip too-early departures or already too-late arrivals
        if dep_ts < earliest_departure - min_origin_buffer:
            continue
        if arr_ts > latest_arrival:
            continue

        # Can we be at u by dep_ts respecting MCT?
        # At origin, use available_from[origin] (may be = earliest_departure if min_origin_buffer=0)
        # Else require earliest[u] + min_connection
        can_board = False
        if u in available_from and dep_ts >= available_from[u]:
            can_board = True
        elif u in earliest and u != origin and dep_ts >= earliest[u] + min_connection:
            can_board = True

        if not can_board:
            continue

        # Relax arrival at v
        if (v not in earliest) or (arr_ts < earliest[v]):
            earliest[v] = arr_ts
            # After arriving at v at arr_ts, the earliest time we can depart from v is arr_ts + min_connection
            available_from[v] = arr_ts + min_connection
            pred[v] = (u, (dep_ts, arr_ts, u, v, carrier, flight_number, row_idx))
            relaxations += 1
    print(
        f"[CSA] Scan finished with {relaxations} relaxations "
        f"in {time.perf_counter() - t_scan:.4f}s"
    )

    # If destination never improved within cutoff
    if (destination not in earliest) or (earliest[destination] > latest_arrival):
        return None, None

    # Reconstruct path
    t_reconstruct = time.perf_counter()
    path: List[Tuple[datetime, datetime, str, str, Optional[str], Optional[str], int]] = []
    a = destination
    while a != origin:
        if a not in pred:
            # No chain back to origin — not reachable
            return None, None
        u, conn = pred[a]
        path.append(conn)
        a = u
    path.reverse()
    print(
        f"[CSA] Path reconstruction produced {len(path)} legs "
        f"in {time.perf_counter() - t_reconstruct:.4f}s"
    )

    # Provide a clean, user-friendly itinerary, including original DataFrame row index
    itinerary = []
    for dep_ts, arr_ts, u, v, carrier, flight_number, row_idx in path:
        r = df.row(row_idx, named=True) if 0 <= row_idx < df.height else None
        itinerary.append({
            "row_idx": row_idx,
            "DEPAPT": u,
            "ARRAPT": v,
            "CARRIER": carrier,
            "FLTNO": flight_number,
            "dep_ts": dep_ts,
            "arr_ts": arr_ts,
            # echo some raw fields back if wanted and available
            "FLIGHT_DATE": r.get("FLIGHT_DATE") if r else None,
            "DEPTIM": r.get("DEPTIM") if r else None,
            "ARRTIM": r.get("ARRTIM") if r else None,
            "ARRDAY": r.get("ARRDAY") if r else None,
            "TOTAL_SEATS": r.get("TOTAL_SEATS") if r else None,
            "ESTIMATED_CO2_TOTAL_TONNES": r.get("ESTIMATED_CO2_TOTAL_TONNES") if r else None,
            "ESTIMATED_CO2_PER_CAPITA": r.get("ESTIMATED_CO2_PER_CAPITA") if r else None,
        })

    return itinerary, earliest[destination]
