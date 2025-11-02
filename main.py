import polars as pl
import os, requests, json, re
from datetime import datetime, timedelta, timezone
import numpy as np
import math

from multi_leg import *
from airports import get_airport_coordinates

# major airports from QRT offices 
OFFICES = [
    "LHR",
    "BOM",
    "CDG",
    "HKG",
    "SIN",
    "UAE",
    "DXB",
    "PVG",
    "ZRH",
    "GVA",
    "AAR",
    "SYD",
    "WRO",
    "BUD"
]

# Start offices
# jump airports
# #

def _clean_numeric(column: str) -> pl.Expr:
    """Return a Float64 cast for columns that may arrive as strings/padded numbers."""
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.replace_all(",", "")
        .cast(pl.Float64, strict=False)
    )

# d = datetime(2024, 1, 20)
# #print(get_flights_score("LHR", "BOM", d, lambda a, b: a*b))

# #print(len(set(emissions["DEPARTURE_AIRPORT"])))

def _ensure_naive(dt):
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

NEARBY_AIRPORTS = {
    "Mumbai": ["BOM", "PNQ", "ISK"],     # BOM: Chhatrapati Shivaji Maharaj International Airport; PNQ: Pune; ISK: Nashik.  [oai_citation:0‡Rome2Rio](https://www.rome2rio.com/s/Nearby-Airports/Mumbai?utm_source=chatgpt.com)
    "Wrocław": ["WRO", "POZ", "KTW"],   # WRO: Copernicus Airport Wrocław; POZ: Poznań; KTW: Katowice.  [oai_citation:1‡airport.globefeed.com](https://airport.globefeed.com/Poland_Nearest_Airport_Result.asp?lat=51.1&lng=17.0333333&place=Wroclaw%2C+Lower+Silesian+Voivodeship%2C+Poland&sr=gp&utm_source=chatgpt.com)
    "London": ["LHR", "LGW", "LCY"],    # LHR: Heathrow; LGW: Gatwick; LCY: London City.
    "New York": ["JFK", "EWR", "LGA"],  # JFK: John F. Kennedy; EWR: Newark; LGA: LaGuardia.
    "Paris": ["CDG", "ORY", "BVA"],     # CDG: Charles de Gaulle; ORY: Orly; BVA: Beauvais.
    "Dubai": ["DXB", "SHJ", "AUH"],     # DXB: Dubai; SHJ: Sharjah; AUH: Abu Dhabi.
    "Zurich": ["ZRH", "BSL", "GVA"],    # ZRH: Zurich; BSL: Basel-Mulhouse; GVA: Geneva.
    "Geneva": ["GVA", "ZRH", "BSL"],    # GVA: Geneva; ZRH & BSL nearby region airports.
    "Budapest": ["BUD", "VIE", "SOB"],  # BUD: Budapest; VIE: Vienna; SOB: (smaller airport near Bratislava).
    "Singapore": ["SIN", "XSP", "JHB"], # SIN: Singapore Changi; XSP: Seletar; JHB: Johor Bahru (Malaysia) 
    "Sydney": ["SYD", "MEL", "BNE"],    # SYD: Sydney; MEL: Melbourne; BNE: Brisbane.
    "Hong Kong": ["HKG", "MFM", "ZUH"], # HKG: Hong Kong; MFM: Macau; ZUH: Zhuhai.
    "Shanghai": ["PVG", "SHA", "HGH"],  # PVG: Shanghai Pudong; SHA: Shanghai Hongqiao; HGH: Hangzhou.
    "Aarhus": ["AAR", "BLL", "CPH"]     # AAR: Aarhus; BLL: Billund; CPH: Copenhagen.
}

CITY_TO_IATA = {
    "Mumbai": "BOM",
    "Shanghai": "PVG",
    "Hong Kong": "HKG",
    "Singapore": "SIN",
    "Sydney": "SYD",
    "London": "LHR",
    "New York": "JFK",
    "Paris": "CDG",
    "Dubai": "DXB",
    "Zurich": "ZRH",
    "Geneva": "GVA",
    "Aarhus": "AAR",
    "Wroclaw": "WRO",
    "Budapest": "BUD",
}

NEARBY_AIRPORTS_BY_IATA = {
    code: tuple(airports)
    for airports in NEARBY_AIRPORTS.values()
    for code in airports
}

EMISSIONS_FILE = "/opt/durhack/emissions.csv"
SCHEDULE_SCHEMA_OVERRIDES = {
    "ARRDAY": pl.Utf8,
}
SCHEDULE_COLUMNS = [
    "FLIGHT_DATE",
    "DEPAPT",
    "ARRAPT",
    "DEPTIM",
    "ARRTIM",
    "ARRDAY",
    "CARRIER",
    "FLTNO",
    "TOTAL_SEATS",
    "ELPTIM",
    "DEPCITY",
    "ARRCITY",
    "INTAPT",
    "SCHEDULED_DEPARTURE_DATE_TIME_UTC",
    "SCHEDULED_ARRIVAL_DATE_TIME_UTC",
]
EMISSIONS_COLUMNS = [
    "CARRIER_CODE",
    "FLIGHT_NUMBER",
    "DEPARTURE_AIRPORT",
    "ARRIVAL_AIRPORT",
    "ESTIMATED_CO2_TOTAL_TONNES",
    "SEATS",
]
CSA_SLACK = timedelta(hours=12)
MAX_NETWORK_HOPS = 1


def _scan_schedule_window(schedule_start, schedule_end):
    schedule_scans = []
    cur = schedule_start
    while cur <= schedule_end:
        schedule_file = (
            f"/opt/durhack/schedules/{cur.year}/"
            f"{cur.month:02d}/{cur.day:02d}.csv"
        )
        if os.path.exists(schedule_file):
            schedule_scans.append(
                pl.scan_csv(
                    schedule_file,
                    infer_schema_length=10000,
                    schema_overrides=SCHEDULE_SCHEMA_OVERRIDES,
                ).with_columns(
                    pl.col("ARRDAY").cast(pl.Utf8, strict=False),
                    pl.col("CARRIER").cast(pl.Utf8, strict=False),
                    pl.col("FLTNO").cast(pl.Utf8, strict=False),
                    pl.col("DEPAPT").cast(pl.Utf8, strict=False),
                    pl.col("ARRAPT").cast(pl.Utf8, strict=False),
                )
            )
        cur += timedelta(days=1)

    if not schedule_scans:
        raise FileNotFoundError(
            f"No schedule files found between {schedule_start.date()} and {schedule_end.date()}"
        )

    return pl.concat(schedule_scans) if len(schedule_scans) > 1 else schedule_scans[0]


def _load_emissions_table():
    return (
        pl.scan_csv(EMISSIONS_FILE, infer_schema_length=10000)
        .select(EMISSIONS_COLUMNS)
        .with_columns(
            pl.col("FLIGHT_NUMBER").cast(pl.Utf8),
            pl.col("CARRIER_CODE").cast(pl.Utf8),
            pl.col("DEPARTURE_AIRPORT").cast(pl.Utf8),
            pl.col("ARRIVAL_AIRPORT").cast(pl.Utf8),
            _clean_numeric("SEATS").alias("SEATS"),
            _clean_numeric("ESTIMATED_CO2_TOTAL_TONNES").alias(
                "ESTIMATED_CO2_TOTAL_TONNES"
            ),
        )
    )


def _prepare_flight_dataframe(schedules_lf):
    emissions = _load_emissions_table()
    return (
        schedules_lf.select(SCHEDULE_COLUMNS)
        .with_columns(
            pl.col("FLTNO").cast(pl.Utf8),
            pl.col("CARRIER").cast(pl.Utf8),
            pl.col("DEPAPT").cast(pl.Utf8),
            pl.col("ARRAPT").cast(pl.Utf8),
            _clean_numeric("TOTAL_SEATS").alias("TOTAL_SEATS"),
        )
        .join(
            emissions,
            left_on=["CARRIER", "FLTNO", "DEPAPT", "ARRAPT"],
            right_on=[
                "CARRIER_CODE",
                "FLIGHT_NUMBER",
                "DEPARTURE_AIRPORT",
                "ARRIVAL_AIRPORT",
            ],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("SEATS").is_not_null() & (pl.col("SEATS") > 0))
            .then(pl.col("SEATS"))
            .otherwise(pl.col("TOTAL_SEATS"))
            .alias("_seat_capacity")
        )
        .with_columns(
            pl.when(
                (pl.col("ESTIMATED_CO2_TOTAL_TONNES").is_not_null())
                & (pl.col("_seat_capacity").is_not_null())
                & (pl.col("_seat_capacity") > 0)
            )
            .then(pl.col("ESTIMATED_CO2_TOTAL_TONNES") / pl.col("_seat_capacity"))
            .otherwise(None)
            .alias("ESTIMATED_CO2_PER_CAPITA")
        )
        .filter(
            pl.col("ESTIMATED_CO2_TOTAL_TONNES").is_not_null()
            & pl.col("_seat_capacity").is_not_null()
            & (pl.col("_seat_capacity") > 0)
        )
        .drop("_seat_capacity")
        .collect()
    )


def _filter_timetable_by_time(df_ts, depart_after, arrive_before, slack=CSA_SLACK):
    dep_lower_bound = depart_after - slack
    dep_upper_bound = arrive_before + slack
    arr_upper_bound = arrive_before + slack
    return df_ts.filter(
        (pl.col("dep_ts") >= pl.lit(dep_lower_bound))
        & (pl.col("dep_ts") <= pl.lit(dep_upper_bound))
        & (pl.col("arr_ts") <= pl.lit(arr_upper_bound))
    )


def _filter_timetable_by_network(df_ts, outbound_map, max_hops=MAX_NETWORK_HOPS):
    focus_airports = set(OFFICES)
    for outbound_office in outbound_map.keys():
        if isinstance(outbound_office, str):
            focus_airports.add(outbound_office)
        focus_airports.update(_candidate_departure_airports(outbound_office))
    network_airports = set(focus_airports)
    for _ in range(max_hops):
        hop_filter = df_ts.filter(
            pl.col("DEPAPT").is_in(network_airports)
            | pl.col("ARRAPT").is_in(network_airports)
        )
        reachable_dep = (
            hop_filter["DEPAPT"].drop_nulls().to_list()
            if "DEPAPT" in hop_filter.columns
            else []
        )
        reachable_arr = (
            hop_filter["ARRAPT"].drop_nulls().to_list()
            if "ARRAPT" in hop_filter.columns
            else []
        )
        hop_airports = set(reachable_dep) | set(reachable_arr)
        before = len(network_airports)
        network_airports |= hop_airports
        if len(network_airports) == before:
            break

    return df_ts.filter(
        pl.col("DEPAPT").is_in(network_airports)
        | pl.col("ARRAPT").is_in(network_airports)
    )


def _candidate_departure_airports(outbound_office):
    """
    Return a de-duplicated, ordered list of potential origin airports for an office.
    """

    def _normalize_iterable(value):
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        if value is None:
            return []
        return [str(value)]

    initial = _normalize_iterable(outbound_office)
    candidates = []

    def _extend(values):
        for val in values or []:
            val = str(val)
            if val and val not in seen:
                seen.add(val)
                candidates.append(val)

    seen = set()
    _extend(initial)

    for code in initial:
        _extend(NEARBY_AIRPORTS_BY_IATA.get(code))
        _extend(CITY_TO_IATA.get(code))

    # Fallback to original office code if nothing else was found
    if not candidates and initial:
        _extend(initial)

    return candidates


def _compute_route_co2(itinerary):
    route_co2_per_capita = 0.0
    for leg in itinerary:
        leg_co2 = leg.get("ESTIMATED_CO2_PER_CAPITA")
        if leg_co2 in (None, ""):
            total_tonnes = leg.get("ESTIMATED_CO2_TOTAL_TONNES")
            seats = leg.get("TOTAL_SEATS")
            try:
                seats_val = float(seats)
            except (TypeError, ValueError):
                seats_val = None
            if seats_val is None or seats_val <= 0:
                seats = leg.get("SEATS")
            try:
                total_val = float(total_tonnes) if total_tonnes is not None else None
                seats_val = float(seats) if seats not in (None, "") else None
                if (
                    total_val is not None
                    and seats_val is not None
                    and math.isfinite(total_val)
                    and math.isfinite(seats_val)
                    and seats_val > 0.0
                ):
                    leg_co2 = total_val / seats_val
                else:
                    leg_co2 = 0.0
            except Exception:
                leg_co2 = 0.0
        try:
            route_co2_per_capita += float(leg_co2)
        except Exception:
            route_co2_per_capita += 0.0
    return route_co2_per_capita


def _build_route_nodes(itinerary):
    if not itinerary:
        return []

    airport_sequence = []
    first_leg = itinerary[0]
    airport_sequence.append(first_leg.get("DEPAPT"))
    for leg in itinerary:
        airport_sequence.append(leg.get("ARRAPT"))

    route_nodes = []
    for code in airport_sequence:
        coords = get_airport_coordinates(code)
        route_nodes.append(
            {
                "airport_code": coords.get("airport_code"),
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
            }
        )
    return route_nodes


def _to_iso_z(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _evaluate_single_meeting_point(
    df_ts, meeting_point, outbound_map, depart_after, arrive_before, duration_days
):
    attendee_hours_by_office = {}
    co2_by_office = {}
    routes_by_office = {}
    stats_by_office = {
        "attendee_travel_hours": attendee_hours_by_office,
        "attendee_co2": co2_by_office,
        "attendee_routes": routes_by_office,
    }

    per_attendee_hours = []
    arrival_times = []
    total_co2 = 0.0
    all_routes_found = True

    for outbound_office, num in outbound_map.items():
        if (meeting_point == outbound_office) or (num <= 0):
            continue
        best_choice = None
        best_co2 = None
        candidate_airports = _candidate_departure_airports(outbound_office)

        for origin_airport in candidate_airports:
            itinerary, eta = fastest_leq_one_stop(
                df_ts,
                origin_airport,
                meeting_point,
                earliest_departure=depart_after,
                latest_arrival=arrive_before,
                min_connection=timedelta(minutes=30),   # set connection buffer
                min_origin_buffer=timedelta(minutes=0),
            )

            if not itinerary or eta is None:
                continue

            journey_time = itinerary_travel_time(itinerary)
            if journey_time is None:
                continue

            route_co2_per_capita = _compute_route_co2(itinerary)

            if best_choice is None or route_co2_per_capita < best_co2:
                best_choice = (itinerary, eta, journey_time, route_co2_per_capita)
                best_co2 = route_co2_per_capita
            elif (
                best_choice is not None
                and math.isclose(route_co2_per_capita, best_co2)
                and journey_time < best_choice[2]
            ):
                # Prefer shorter travel time when CO2 impact is effectively tied.
                best_choice = (itinerary, eta, journey_time, route_co2_per_capita)
                best_co2 = route_co2_per_capita

        if not best_choice:
            all_routes_found = False
            break

        itinerary, eta, journey_time, route_co2_per_capita = best_choice

        hours = journey_time.total_seconds() / 3600
        attendee_hours_by_office[outbound_office] = round(hours, 2)
        per_attendee_hours.extend([hours] * num)
        arrival_times.extend([eta] * num)

        total_co2 += route_co2_per_capita * num
        co2_by_office[outbound_office] = {
            "per_attendee": round(route_co2_per_capita, 3),
            "total": round(route_co2_per_capita * num, 3),
        }

        routes_by_office[outbound_office] = _build_route_nodes(itinerary)

    for outbound_office in outbound_map:
        co2_by_office.setdefault(
            outbound_office,
            {"per_attendee": 0.0, "total": 0.0},
        )
        attendee_hours_by_office.setdefault(outbound_office, 0.0)
        routes_by_office.setdefault(outbound_office, [])

    if not all_routes_found or not per_attendee_hours or not arrival_times:
        return (
            stats_by_office,
            {
                "event_dates": {"start": None, "end": None},
                "event_span": {"start": None, "end": None},
                "total_co2": 0.0,
                "average_travel_hours": 0.0,
                "median_travel_hours": 0.0,
                "max_travel_hours": 0.0,
                "min_travel_hours": 0.0,
                "fairness": 0.0,
                "total_score": 0.0,
            },
        )

    avg = float(np.mean(per_attendee_hours))
    med = float(np.median(per_attendee_hours))
    mx = float(np.max(per_attendee_hours))
    mn = float(np.min(per_attendee_hours))
    fairness = float(np.var(per_attendee_hours)) if len(per_attendee_hours) > 1 else 0.0

    latest_arrival = max(arrival_times)
    earliest_arrival = min(arrival_times)
    event_start = latest_arrival
    event_end = event_start + timedelta(days=max(duration_days, 0))
    event_span_start = earliest_arrival
    event_span_end = event_end + timedelta(hours=mx)

    co2_value = float(total_co2)
    total_score = 0.5 * fairness + 0.5 * co2_value

    meeting_stats = {
        "event_dates": {
            "start": _to_iso_z(event_start),
            "end": _to_iso_z(event_end),
        },
        "event_span": {
            "start": _to_iso_z(event_span_start),
            "end": _to_iso_z(event_span_end),
        },
        "total_co2": round(co2_value, 3),
        "average_travel_hours": round(avg, 2),
        "median_travel_hours": round(med, 2),
        "max_travel_hours": round(mx, 2),
        "min_travel_hours": round(mn, 2),
        "fairness": round(fairness, 4),
        "total_score": round(total_score, 3),
    }

    return stats_by_office, meeting_stats


def evaluate_naive_atOffice(

    outbound_map,  # outbound_office: num coming from there
    window_start, # dict - year, month, day
    window_end, 
    duration_days):
    print("DUR", duration_days)

    window_start = _ensure_naive(window_start)
    window_end = _ensure_naive(window_end)

    depart_after = window_start + timedelta(days=2)
    arrive_before = window_end + timedelta(days=-duration_days)
    schedule_start = window_start - timedelta(days=2)
    schedule_end = arrive_before

    schedules = _scan_schedule_window(schedule_start, schedule_end)
    flight_date_start = schedule_start.date()
    flight_date_end = schedule_end.date()
    schedules = schedules.filter(
        pl.col("FLIGHT_DATE")
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .is_between(flight_date_start, flight_date_end, closed="both")
    )

    df = _prepare_flight_dataframe(schedules)
    df_ts = with_connection_timestamps(df)
    df_ts = _filter_timetable_by_time(df_ts, depart_after, arrive_before)
    df_ts = _filter_timetable_by_network(df_ts, outbound_map)
    stats_by_meeting_point = {}
    stats_by_meeting_point_by_office = {}

    for meeting_point in OFFICES:
        stats_by_office, meeting_stats = _evaluate_single_meeting_point(
            df_ts,
            meeting_point,
            outbound_map,
            depart_after,
            arrive_before,
            duration_days,
        )
        stats_by_meeting_point_by_office[meeting_point] = stats_by_office
        stats_by_meeting_point[meeting_point] = meeting_stats

    return stats_by_meeting_point, stats_by_meeting_point_by_office
