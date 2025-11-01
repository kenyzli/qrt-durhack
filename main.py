import polars as pl
import os, requests, json, re
from datetime import datetime
import numpy as np


# major airports from QRT offices 
OFFICES = [
    "LHR",
    "BOM",
]

emissions_file = "emissions.csv"
emissions = pl.scan_csv(emissions_file, infer_schema_length=0).collect()
# print(emissions.columns)


# score lower is better
# from A to B on this date. 
# A and B are airport codes like LHR and BOM
def get_flights_score(A: str, B: str, 
                      depart_on: datetime,
                      scoreFunction: callable
                      ) -> pl.DataFrame:
    schedule_file = "01.csv"
    schedules = pl.scan_csv(schedule_file, infer_schema_length=10000)
    emissions = pl.scan_csv(emissions_file, infer_schema_length=10000)
    #
    # print(pl.DataFrame.collect_schema(schedules)) # TOTAL_SEATS - total seats column
    # print(pl.DataFrame.collect_schema(emissions)) # ESTIMATED_CO2_TOTAL_TONNES
    
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
    for day in range(0, (window_end - window_start).days - duration_days + 1):
        # each office gets a score
        # scores = {location: -1 for location in OFFICES}
        final_flights = {} 
        
        # for each possible arrival day, what is the score? 
        # should be optimised. 
        for cur_office in OFFICES:
            cur_score_variance = -1
            flights = {}
            for out_dest, n_out in outbound_map.items(): 
                if out_dest == cur_office:
                    continue
                flights[out_dest] = dict(get_flights_score(
                    out_dest, 
                    cur_office, 
                    window_start,
                    lambda a, b: a*b
                ))
                flights[out_dest]["FLIGHT_SCORE"] *= outbound_map[out_dest] # multiply by number of people
                        
            # if total score from co2 and fairness is better: 
            var =  np.var([f["FLIGHT_SCORE"][0] for f in flights.values()])
            if (cur_score_variance == -1 or cur_score_variance > var):
                cur_score_variance = var
                final_flights = flights
        
        return final_flights # they will be the final flights to take



    
outbound_map = {
    "LHR": 10,
    "BOM": 5
}

window_start = datetime(2024, 1, 20)

window_end = datetime(2024, 1, 25)

# print(get_flights_score("BOM","LHR", window_start, lambda a, b: a*b))


print(evaluate_naive_atOffice(
    outbound_map,  # location: num coming from there
    window_start, # dict - year, month, day
    window_end, 
    0
))