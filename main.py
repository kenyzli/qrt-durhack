import polars as pl
import os, requests, json, re
from datetime import datetime, timedelta
import numpy as np


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

emissions_file = "/opt/durhack/emissions.csv"
emissions = pl.scan_csv(emissions_file, infer_schema_length=10000)

def get_flights_score(A: str, B: str, 
                      depart_on: datetime,
                      scoreFunction: callable
                      ) -> pl.DataFrame:
    schedule_file = (
    f"/opt/durhack/schedules/{depart_on.year}/"
    f"{depart_on.month:02d}/{depart_on.day:02d}.csv"
    )   
    schedules = pl.scan_csv(schedule_file, infer_schema_length=0).collect()
    #
    # print(pl.DataFrame.collect_schema(schedules)) # TOTAL_SEATS - total seats column
    # print(pl.DataFrame.collect_schema(emissions)) # ESTIMATED_CO2_TOTAL_TONNES
    
    # information on the columns in the data files can be found at:
    # https://knowledge.oag.com/docs/wdf-record-layout
    # https://knowledge.oag.com/docs/emissions-schedules-data-fields-explained
    
    flights = (
        schedules.filter((pl.col("DEPAPT") == A) & (pl.col("ARRAPT") == B))
    )
    
    print("Flights:")
    print(flights.collect().columns)
    
    # Join with the emissions data on carrier and flight number, sorting on emissions
    result = (
        flights.join(
            emissions,
            left_on=["CARRIER", "FLTNO"],
            right_on=["CARRIER_CODE", "FLIGHT_NUMBER"],
            how="inner"
        )
        .with_columns(
            (pl.col("ESTIMATED_CO2_TOTAL_TONNES") / pl.col("TOTAL_SEATS").cast(pl.Float64))
            .alias("ESTIMATED_CO2_PER_CAPITA")
        )
        .filter(
            pl.col("ESTIMATED_CO2_PER_CAPITA").is_not_null() & 
            pl.col("ELPTIM").is_not_null()
        )
        .with_columns(
            pl.struct(["ESTIMATED_CO2_PER_CAPITA", "ELPTIM"])
            .map_elements(
                lambda row: scoreFunction(row["ESTIMATED_CO2_PER_CAPITA"], row["ELPTIM"]) 
                if row["ESTIMATED_CO2_PER_CAPITA"] is not None and row["ELPTIM"] is not None 
                else None,
                return_dtype=pl.Float64
            )
        ) 
        .with_columns(
            pl.struct(["ESTIMATED_CO2_PER_CAPITA", "ELPTIM"])
            .map_elements(
                lambda row: scoreFunction(row["ESTIMATED_CO2_PER_CAPITA"], row["ELPTIM"]),
                return_dtype=pl.Float64
            )
            .alias("FLIGHT_SCORE")
        )
        .select([
            pl.col("FLTNO"), 
            pl.col("DEPCITY"),
            pl.col("ARRCITY"),
            
            # pl.col("DEPAPT"),
            # pl.col("ARRAPT"),
            
            pl.col("ARRTIM"),
            pl.col("DEPTIM"),
            pl.col("ELPTIM"),
            # pl.col("ESTIMATED_CO2_TOTAL_TONNES"),
            pl.col("ESTIMATED_CO2_PER_CAPITA"),
            pl.col("FLIGHT_SCORE"),
            pl.col("INTAPT")
        ])
        .sort("FLIGHT_SCORE")
    ).collect()

    return result.head(1).to_dict(as_series=False)

    
# score lower is better
# from A to B on this date. 
# A and B are airport codes like LHR and BOM
def get_flights_score_v2(A: str, B: str, schedules,
                      scoreFunction: callable
                      ) -> pl.DataFrame:
    
    # information on the columns in the data files can be found at:
    # https://knowledge.oag.com/docs/wdf-record-layout
    # https://knowledge.oag.com/docs/emissions-schedules-data-fields-explained
    
    flights = (
        schedules.filter((pl.col("DEPAPT") == A) & (pl.col("ARRAPT") == B))
    )
    # print("Flights:")
    # print(flights.collect().columns)
    
    # Join with the emissions data on carrier and flight number, sorting on emissions
    result = (
        flights.join(
            emissions,
            left_on=["CARRIER", "FLTNO"],
            right_on=["CARRIER_CODE", "FLIGHT_NUMBER"],
            how="inner"
        )
        .with_columns(
            (pl.col("ESTIMATED_CO2_TOTAL_TONNES") / pl.col("TOTAL_SEATS").cast(pl.Float64))
            .alias("ESTIMATED_CO2_PER_CAPITA")
        )
        .filter(
            pl.col("ESTIMATED_CO2_PER_CAPITA").is_not_null() & 
            pl.col("ELPTIM").is_not_null()
        )
        .with_columns(
            pl.struct(["ESTIMATED_CO2_PER_CAPITA", "ELPTIM"])
            .map_elements(
                lambda row: scoreFunction(row["ESTIMATED_CO2_PER_CAPITA"], row["ELPTIM"]) 
                if row["ESTIMATED_CO2_PER_CAPITA"] is not None and row["ELPTIM"] is not None 
                else None,
                return_dtype=pl.Float64
            )
        ) 
        .with_columns(
            pl.struct(["ESTIMATED_CO2_PER_CAPITA", "ELPTIM"])
            .map_elements(
                lambda row: scoreFunction(row["ESTIMATED_CO2_PER_CAPITA"], row["ELPTIM"]),
                return_dtype=pl.Float64
            )
            .alias("FLIGHT_SCORE")
        )
        .select([
            pl.col("FLTNO"), 
            pl.col("DEPCITY"),
            pl.col("ARRCITY"),
            
            # pl.col("DEPAPT"),
            # pl.col("ARRAPT"),
            
            pl.col("ARRTIM"),
            pl.col("DEPTIM"),
            pl.col("ELPTIM"),
            # pl.col("ESTIMATED_CO2_TOTAL_TONNES"),
            pl.col("ESTIMATED_CO2_PER_CAPITA"),
            pl.col("FLIGHT_SCORE"),
            pl.col("INTAPT")
        ])
        .sort("FLIGHT_SCORE")
    ).collect()

    return result.head(1).to_dict(as_series=False)

# d = datetime(2024, 1, 20)
# print(get_flights_score("LHR", "BOM", d, lambda a, b: a*b))

# print(len(set(emissions["DEPARTURE_AIRPORT"])))

def evaluate_naive_atOffice(
    outbound_map,  # location: num coming from there
    window_start, # dict - year, month, day
    window_end, 
    duration_days
):
    # print((window_end - window_start).days - duration_days + 1)
    for day in range(0, (window_end - window_start).days - duration_days):
        # each office gets a score
        # scores = {location: -1 for location in OFFICES}
        final_flights = {} 
        depart_on = window_start + timedelta(days=day)
        schedule_file = (
        f"/opt/durhack/schedules/{depart_on.year}/"
        f"{depart_on.month:02d}/{depart_on.day:02d}.csv"
        )   
        schedules = pl.scan_csv(schedule_file, infer_schema_length=10000)
        
        # for each possible arrival day, what is the score? 
        # should be optimised. 
        for cur_office in OFFICES:
            cur_score_variance = -1
            cur_total = -1
            flights = {}
            for out_from, n_out in outbound_map.items(): 
                # print("doing", out_from, "to", cur_office)
                if out_from == cur_office:
                    continue
                res = dict(get_flights_score_v2(
                    out_from, 
                    cur_office, 
                    schedules,
                    lambda a, b: a*b
                ))
                if res["FLTNO"]:
                    flights[out_from] = res
                    flights[out_from]["FLIGHT_SCORE"] *= outbound_map[out_from] # multiply by number of people
                        
            # if total score from co2 and fairness is better: 
            score_ls = [f["FLIGHT_SCORE"][0] for f in flights.values()]
            var =  np.var(score_ls)
            total = sum(score_ls)
            # if (cur_score_variance == -1 or cur_score_variance > var):
            if cur_total == -1 or cur_total > total:
                cur_score_variance = var
                final_flights = flights
        
        return final_flights # they will be the final flights to take



    
outbound_map = {
    "LHR": 10,
    "BOM": 5
}

window_start = datetime(2024, 1, 20)
window_end = datetime(2024, 1, 25)

depart_on = window_start
schedule_file = (
    f"/opt/durhack/schedules/{depart_on.year}/"
    f"{depart_on.month:02d}/{depart_on.day:02d}.csv"
)   
schedules = pl.scan_csv(schedule_file, infer_schema_length=10000)

# print(get_flights_score_v2("BOM", "LHR", schedules, lambda a, b: a*b))
# print(get_flights_score("BOM","UAE", window_start, lambda a, b: a*b))


print(evaluate_naive_atOffice(
    outbound_map,  # location: num coming from there
    window_start, # dict - year, month, day
    window_end, 
    0
))