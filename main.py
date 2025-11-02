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

def evaluate_naive_atOffice(

    outbound_map,  # outbound_office: num coming from there
    window_start, # dict - year, month, day
    window_end, 
    duration_days):
    print("DUR", duration_days)

    window_start = _ensure_naive(window_start)
    window_end = _ensure_naive(window_end)

    # find fastest that arrives beofore window_end - duration_days
    # ^ for each possible meeting
    # calculate cost for legs
    # sum costs
    # take min
    # #print((window_end - window_start).days - duration_days + 1)

    depart_after = window_start + timedelta(days=2)
    arrive_before = window_end + timedelta(days=-duration_days)
    emissions_file = "/opt/durhack/emissions.csv"
    emissions = pl.scan_csv(emissions_file, infer_schema_length=10000)
    SCHEDULE_SCHEMA_OVERRIDES = {
        "ARRDAY": pl.Utf8,
    }
    # gather all csv in window - 2 to window-duration_days
    schedule_start = window_start - timedelta(days=2)
    schedule_end = arrive_before

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
                )
                .with_columns(
                    pl.col("ARRDAY").cast(pl.Utf8, strict=False),
                    pl.col("CARRIER").cast(pl.Utf8, strict=False),
                    pl.col("FLTNO").cast(pl.Utf8, strict=False),
                    pl.col("DEPAPT").cast(pl.Utf8, strict=False),
                    pl.col("ARRAPT").cast(pl.Utf8, strict=False),
                    pl.col("SCHEDULED_DEPARTURE_DATE_TIME_UTC").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.3f", strict=False),
                    pl.col("SCHEDULED_ARRIVAL_DATE_TIME_UTC").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.3f", strict=False),
                    pl.col("SCHEDULED_DEPARTURE_DATE_TIME_LOCAL").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.3f", strict=False),
                    pl.col("SCHEDULED_ARRIVAL_DATE_TIME_LOCAL").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.3f", strict=False),

                )
            )
        else:
            pass
            #print(f"Warning: missing schedule file for {cur.date()} at {schedule_file}")
        cur += timedelta(days=1)



    if not schedule_scans:
        raise FileNotFoundError(
            f"No schedule files found between {schedule_start.date()} and {schedule_end.date()}"
        )

    schedules = (
        pl.concat(schedule_scans) if len(schedule_scans) > 1 else schedule_scans[0]
    )

    # Restrict to flights operating inside the schedule window we loaded from disk.
    flight_date_start = schedule_start.date()
    flight_date_end = schedule_end.date()
    schedules = schedules.filter(
        pl.col("FLIGHT_DATE")
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .is_between(flight_date_start, flight_date_end, closed="both") &
        pl.col('GHOST').is_not_null()
    )

    schedule_cols = [
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
        "SCHEDULED_DEPARTURE_DATE_TIME_LOCAL",
        "SCHEDULED_ARRIVAL_DATE_TIME_LOCAL"
    ]

    emissions_cols = [
        "CARRIER_CODE",
        "FLIGHT_NUMBER",
        "DEPARTURE_AIRPORT",
        "ARRIVAL_AIRPORT",
        "ESTIMATED_CO2_TOTAL_TONNES",
        "SEATS",
    ]

    df = (
        schedules
        .select(schedule_cols)
        .with_columns(
            pl.col("FLTNO").cast(pl.Utf8),
            pl.col("CARRIER").cast(pl.Utf8),
            pl.col("DEPAPT").cast(pl.Utf8),
            pl.col("ARRAPT").cast(pl.Utf8),
            _clean_numeric("TOTAL_SEATS").alias("TOTAL_SEATS"),
        )
        .with_columns(
            # Offset in hours: difference between local and UTC
            ((pl.col("SCHEDULED_DEPARTURE_DATE_TIME_LOCAL") - pl.col("SCHEDULED_DEPARTURE_DATE_TIME_UTC"))
                .dt.total_seconds() / 3600).alias("DEPARTURE_TZ_OFFSET_HOURS"),

            ((pl.col("SCHEDULED_ARRIVAL_DATE_TIME_LOCAL") - pl.col("SCHEDULED_ARRIVAL_DATE_TIME_UTC"))
                .dt.total_seconds() / 3600).alias("ARRIVAL_TZ_OFFSET_HOURS"),

            # Timezone change / jetlag direction (positive = eastbound)
            ((pl.col("SCHEDULED_ARRIVAL_DATE_TIME_LOCAL") - pl.col("SCHEDULED_ARRIVAL_DATE_TIME_UTC")).dt.total_seconds()/3600
                - (pl.col("SCHEDULED_DEPARTURE_DATE_TIME_LOCAL") - pl.col("SCHEDULED_DEPARTURE_DATE_TIME_UTC")).dt.total_seconds()/3600).alias("TIMEZONE_SHIFT_HOURS"))
                .join(
            emissions.select(emissions_cols).with_columns(
                pl.col("FLIGHT_NUMBER").cast(pl.Utf8),
                pl.col("CARRIER_CODE").cast(pl.Utf8),
                pl.col("DEPARTURE_AIRPORT").cast(pl.Utf8),
                pl.col("ARRIVAL_AIRPORT").cast(pl.Utf8),
                _clean_numeric("SEATS").alias("SEATS"),
                _clean_numeric("ESTIMATED_CO2_TOTAL_TONNES").alias("ESTIMATED_CO2_TOTAL_TONNES"),
            ),
            left_on=["CARRIER", "FLTNO", "DEPAPT", "ARRAPT"],
            right_on=["CARRIER_CODE", "FLIGHT_NUMBER", "DEPARTURE_AIRPORT", "ARRIVAL_AIRPORT"],
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

    df_ts = with_connection_timestamps(df)

    # Trim the timetable before running CSA so we don't scan hundreds of thousands of irrelevant flights.
    # Allow a small slack either side of the computed travel window to keep viable connections.
    csa_slack = timedelta(hours=12)
    dep_lower_bound = depart_after - csa_slack
    dep_upper_bound = arrive_before + csa_slack
    arr_upper_bound = arrive_before + csa_slack
    pre_filter_rows = df_ts.height
    df_ts = df_ts.filter(
        (pl.col("dep_ts") >= pl.lit(dep_lower_bound))
        & (pl.col("dep_ts") <= pl.lit(dep_upper_bound))
        & (pl.col("arr_ts") <= pl.lit(arr_upper_bound))
    )
    #print(
        # f"[CSA] Time-window filter reduced timetable from {pre_filter_rows} to {df_ts.height} rows "
        # f"for window {dep_lower_bound} → {arr_upper_bound}"
    # )

    # Further prune to airports that are actually reachable from our attendees within a few legs.
    # This avoids carrying flights that are completely disconnected from any origin/meeting point.
    focus_airports = set(outbound_map.keys()) | set(OFFICES)
    network_airports = set(focus_airports)
    max_hops = 1 # allow up to 3 hops (4 legs) to expand the reachable network
    for hop in range(max_hops):
        hop_filter = df_ts.filter(
            pl.col("DEPAPT").is_in(network_airports)
            | pl.col("ARRAPT").is_in(network_airports)
        )
        if "DEPAPT" in hop_filter.columns:
            reachable_dep = hop_filter["DEPAPT"].drop_nulls().to_list()
        else:
            reachable_dep = []
        if "ARRAPT" in hop_filter.columns:
            reachable_arr = hop_filter["ARRAPT"].drop_nulls().to_list()
        else:
            reachable_arr = []
        hop_airports = set(reachable_dep) | set(reachable_arr)
        before = len(network_airports)
        network_airports |= hop_airports
        #print(
        #     f"[CSA] Hop {hop + 1}: expanded network to {len(network_airports)} airports"
        # )
        if len(network_airports) == before:
            break

    network_filter_rows = df_ts.height
    df_ts = df_ts.filter(
        pl.col("DEPAPT").is_in(network_airports)
        | pl.col("ARRAPT").is_in(network_airports)
    )
    #print(
        # f"[CSA] Network filter reduced timetable from {network_filter_rows} to {df_ts.height} rows "
        # f"across {len(network_airports)} airports"
    # )

    stats_by_meeting_point = {}
    stats_by_meeting_point_by_office = {}

    def _to_iso_z(dt):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    for meeting_point in OFFICES:
        attendee_hours_by_office = {}
        co2_by_office = {}
        routes_by_office = {}
        stats_by_meeting_point_by_office[meeting_point] = {
            "attendee_travel_hours": attendee_hours_by_office,
            "attendee_co2": co2_by_office,
            "attendee_routes": routes_by_office,
            "attendee_timezone_shift": attendee_timezone_by_office,
        }
        attendee_timezone_by_office = {}
        per_attendee_hours = []
        per_attendee_timezone_shifts = []
        arrival_times = []
        total_co2 = 0.0
        all_routes_found = True

        for outbound_office, num in outbound_map.items():
            if (meeting_point == outbound_office) or (num <= 0):
                continue
            itinerary, eta = fastest_leq_one_stop(
                df_ts,
                outbound_office,
                meeting_point,
                earliest_departure=depart_after,
                latest_arrival=arrive_before,
                min_connection=timedelta(minutes=30),   # set connection buffer
                min_origin_buffer=timedelta(minutes=0),
            )

            if not itinerary or eta is None:
                #print(f"❌ No valid route from {outbound_office} to {meeting_point} before {arrive_before}")
                all_routes_found = False
                break

            journey_time = itinerary_travel_time(itinerary)
            if journey_time is None:
                #print(f"⚠️ Unable to determine journey time for {outbound_office} → {meeting_point}")
                all_routes_found = False
                break

            hours = journey_time.total_seconds() / 3600
            attendee_hours_by_office[outbound_office] = round(hours, 2)
            per_attendee_hours.extend([hours] * num)
            per_attendee_timezone_shifts = []
            arrival_times.extend([eta] * num)
            route_tz_shift = 0.0
            route_co2_per_capita = 0.0
            for leg in itinerary:
                try:
                    route_tz_shift += float(leg.get("TIMEZONE_SHIFT_HOURS") or 0.0)
                except Exception:
                    pass
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
            attendee_timezone_by_office[outbound_office] = round(route_tz_shift, 2)
            per_attendee_timezone_shifts.extend([route_tz_shift] * num)
            total_co2 += route_co2_per_capita * num
            co2_by_office[outbound_office] = {
                "per_attendee": round(route_co2_per_capita, 3),
                "total": round(route_co2_per_capita * num, 3),
            }

            airport_sequence = []
            if itinerary:
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
            routes_by_office[outbound_office] = route_nodes

            #print(f"✅ Fastest route {outbound_office} → {meeting_point}, ETA {eta}")
            for i, leg in enumerate(itinerary, 1):
                dep = leg["dep_ts"].strftime("%Y-%m-%d %H:%M")
                arr = leg["arr_ts"].strftime("%Y-%m-%d %H:%M")
                #print(f"{i:02d}. {leg['DEPAPT']} → {leg['ARRAPT']}  {dep} → {arr}")
            #print(f"Total travel time from {outbound_office}: {journey_time} (~{hours:.2f} hours)")

        for outbound_office in outbound_map:
            co2_by_office.setdefault(
                outbound_office,
                {"per_attendee": 0.0, "total": 0.0},
            )
            attendee_hours_by_office.setdefault(outbound_office, 0.0)
            attendee_timezone_by_office.setdefault(outbound_office, 0.0)
            routes_by_office.setdefault(outbound_office, [])

        if not all_routes_found or not per_attendee_hours or not arrival_times:
            stats_by_meeting_point[meeting_point] = {
                "event_dates": {"start": None, "end": None},
                "event_span": {"start": None, "end": None},
                "total_co2": 0.0,
                "average_travel_hours": 0.0,
                "median_travel_hours": 0.0,
                "max_travel_hours": 0.0,
                "min_travel_hours": 0.0,
                "fairness": 0.0,
                "total_score": 0.0,
            }
            continue

        avg = float(np.mean(per_attendee_hours))
        med = float(np.median(per_attendee_hours))
        mx = float(np.max(per_attendee_hours))
        mn = float(np.min(per_attendee_hours))
        fairness = float(np.var(per_attendee_hours)) if len(per_attendee_hours) > 1 else 0.0
        tz_avg = float(np.mean(per_attendee_timezone_shifts)) if per_attendee_timezone_shifts else 0.0
        tz_med = float(np.median(per_attendee_timezone_shifts)) if per_attendee_timezone_shifts else 0.0
        tz_mx = float(np.max(per_attendee_timezone_shifts)) if per_attendee_timezone_shifts else 0.0
        tz_mn = float(np.min(per_attendee_timezone_shifts)) if per_attendee_timezone_shifts else 0.0
        latest_arrival = max(arrival_times)
        earliest_arrival = min(arrival_times)
        event_start = latest_arrival
        event_end = event_start + timedelta(days=max(duration_days, 0))
        event_span_start = earliest_arrival
        event_span_end = event_end + timedelta(hours=mx)

        co2_value = float(total_co2)
        total_score = 0.5 * fairness + 0.5 * co2_value + 0 * tz_avg

        stats_by_meeting_point[meeting_point] = {
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
            "average_timezone_shift_hours": round(tz_avg, 2),
            "median_timezone_shift_hours": round(tz_med, 2),
            "max_timezone_shift_hours": round(tz_mx, 2),
            "min_timezone_shift_hours": round(tz_mn, 2),
            "fairness": round(fairness, 4),
            "total_score": round(total_score, 3),
        }

    return stats_by_meeting_point, stats_by_meeting_point_by_office
