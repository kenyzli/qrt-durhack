from __future__ import annotations
import polars as pl
from datetime import datetime, timedelta
from math import inf
from typing import Optional, List, Dict, Tuple

# ---------- Timestamp helpers for OAG-like HHMM + ARRDAY ----------

def _hhmm_to_timedelta(hhmm: int) -> timedelta:
    """Convert HHMM integer (e.g., 5 -> 00:05, 10 -> 00:10, 1650 -> 16:50) to timedelta."""
    h, m = divmod(int(hhmm), 100)
    return timedelta(hours=h, minutes=m)

def _arrday_to_offset(arrday: Optional[str]) -> int:
    """
    Map ARRDAY markers to day offsets:
      'P' => -1  (arrives previous day)
      '1' => +1
      '2' => +2
      None or '' or '0' or unrecognised => 0
    """
    if arrday is None:
        return 0
    arrday = str(arrday).strip().upper()
    if arrday == "P":
        return -1
    if arrday.isdigit():
        return int(arrday)
    return 0

def with_connection_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """
    Input columns (strings/ints as specified):
      - DEPAPT (str), ARRAPT (str)
      - FLIGHT_DATE (YYYY-MM-DD str)
      - DEPTIM (int HHMM)
      - ARRTIM (int HHMM)
      - ARRDAY (str marker: 'P', '1', '2', ... or empty)
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
            (
                pl.col("dep_date")
                .cast(pl.Datetime)
                + (pl.col("DEPTIM").cast(pl.Int64).map_elements(_hhmm_to_timedelta))
            ).alias("dep_ts"),
            # Arrival date = dep_date + day_offset(ARRDAY)
            (
                pl.col("dep_date")
                + pl.col("ARRDAY_norm").map_elements(lambda s: timedelta(days=_arrday_to_offset(s)))
            ).alias("arr_date"),
        )
        .with_columns(
            # Arrival timestamp = arrival_date + HHMM
            (
                pl.col("arr_date").cast(pl.Datetime)
                + (pl.col("ARRTIM").cast(pl.Int64).map_elements(_hhmm_to_timedelta))
            ).alias("arr_ts")
        )
        .drop(["dep_date", "arr_date", "ARRDAY_norm"])
    )
    return out

# ---------- Connection Scan Algorithm (CSA) ----------

class Connection(Tuple):
    __slots__ = ()
    # Just documenting structure:
    # (dep_ts, arr_ts, dep_airport, arr_airport, row_idx)

def build_connections(df: pl.DataFrame) -> List[Tuple[datetime, datetime, str, str, int]]:
    """
    Convert to a list of connections sorted by departure time.
    Each connection is a tuple: (dep_ts, arr_ts, dep_airport, arr_airport, row_idx)
    """
    # Select only the needed columns to minimize Python overhead
    slim = df.select(
        "dep_ts", "arr_ts", "DEPAPT", "ARRAPT"
    ).with_row_index(name="__row_idx__")

    # Bring to Python lists (fast; avoids per-row .to_dict cost)
    dep_ts = slim["dep_ts"].to_list()
    arr_ts = slim["arr_ts"].to_list()
    dep_ap = slim["DEPAPT"].to_list()
    arr_ap = slim["ARRAPT"].to_list()
    idxs   = slim["__row_idx__"].to_list()

    conns = list(zip(dep_ts, arr_ts, dep_ap, arr_ap, idxs))
    conns.sort(key=lambda x: x[0])  # sort by departure time
    return conns

def csa_fastest_route(
    df: pl.DataFrame,
    origin: str,
    destination: str,
    earliest_departure: datetime,
    latest_arrival: datetime,
    min_connection: timedelta = timedelta(minutes=30),
    min_origin_buffer: timedelta = timedelta(0),
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
    window = df.filter(
        (pl.col("arr_ts") <= pl.lit(latest_arrival))
        & (pl.col("dep_ts") >= pl.lit(earliest_departure - min_origin_buffer))
    )

    if window.height == 0:
        return None, None

    # Build scan list
    conns = build_connections(window)

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
    pred: Dict[str, Tuple[str, Tuple[datetime, datetime, str, str, int]]] = {}

    # Scan
    for dep_ts, arr_ts, u, v, row_idx in conns:
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
            pred[v] = (u, (dep_ts, arr_ts, u, v, row_idx))

    # If destination never improved within cutoff
    if (destination not in earliest) or (earliest[destination] > latest_arrival):
        return None, None

    # Reconstruct path
    path: List[Tuple[datetime, datetime, str, str, int]] = []
    a = destination
    while a != origin:
        if a not in pred:
            # No chain back to origin — not reachable
            return None, None
        u, conn = pred[a]
        path.append(conn)
        a = u
    path.reverse()

    # Provide a clean, user-friendly itinerary, including original DataFrame row index
    itinerary = []
    for dep_ts, arr_ts, u, v, row_idx in path:
        r = df.row(row_idx, named=True) if 0 <= row_idx < df.height else None
        itinerary.append({
            "row_idx": row_idx,
            "DEPAPT": u,
            "ARRAPT": v,
            "dep_ts": dep_ts,
            "arr_ts": arr_ts,
            # echo some raw fields back if wanted and available
            "FLIGHT_DATE": r.get("FLIGHT_DATE") if r else None,
            "DEPTIM": r.get("DEPTIM") if r else None,
            "ARRTIM": r.get("ARRTIM") if r else None,
            "ARRDAY": r.get("ARRDAY") if r else None,
        })

    return itinerary, earliest[destination]

# ---------- Example usage ----------

schedule_file  = "durhack-2025-01/01.csv"
emissions_file = "emissions.csv"  # keep if you’ll enrich later

schedules = pl.scan_csv(schedule_file, infer_schema_length=10000)
emissions = pl.scan_csv(emissions_file, infer_schema_length=10000)
cols = ["FLIGHT_DATE", "DEPAPT", "ARRAPT", "DEPTIM", "ARRTIM", "ARRDAY"]

df = (
    schedules
    .select(cols)
    .collect()  # materialise from lazy
)

print("Loaded", df.height, "rows")

# ------------------------------
# 3. Add timestamps (from earlier helper)
# ------------------------------

# Paste in the functions from previous message:
# - _hhmm_to_timedelta
# - _arrday_to_offset
# - with_connection_timestamps
# - csa_fastest_route

df_ts = with_connection_timestamps(df)

# ------------------------------
# 4. Example search query
# ------------------------------

origin = "HND"        # starting airport
destination = "SIN"   # ending airport
t0 = datetime(2025, 1, 1, 6, 0)          # earliest departure
t_deadline = datetime(2025, 1, 3, 6, 0)  # must arrive before this

itinerary, eta = csa_fastest_route(
    df_ts,
    origin,
    destination,
    earliest_departure=t0,
    latest_arrival=t_deadline,
    min_connection=timedelta(minutes=45),   # set connection buffer
    min_origin_buffer=timedelta(minutes=0),
)

# ------------------------------
# 5. Display results
# ------------------------------

if itinerary:
    print(f"✅ Fastest route {origin} → {destination}, ETA {eta}")
    for i, leg in enumerate(itinerary, 1):
        dep = leg['dep_ts'].strftime("%Y-%m-%d %H:%M")
        arr = leg['arr_ts'].strftime("%Y-%m-%d %H:%M")
        print(f"{i:02d}. {leg['DEPAPT']} → {leg['ARRAPT']}  {dep} → {arr}")
else:
    print(f"❌ No valid route from {origin} to {destination} before {t_deadline}")