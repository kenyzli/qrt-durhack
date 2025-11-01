import polars as pl
import os, requests, json, re

# major airports from QRT offices 
OFFICES = [
    "LHR"
]

emissions_file = "/opt/durhack/emissions.csv"
emissions = pl.scan_csv(emissions_file, infer_schema_length=0).collect()
# print(emissions.columns)

# from A to B on this date. 
# A and B are airport codes like LHR and BOM
def get_flights_score(A: str, B: str, year: int, month: int, day: int) -> pl.DataFrame:
    schedule_file = f"/opt/durhack/schedules/{year}/{month}/{day}.csv"
    
    schedules = pl.scan_csv(schedule_file, infer_schema_length=10000)
    emissions = pl.scan_csv(emissions_file, infer_schema_length=10000)
    
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
        .sort("ESTIMATED_CO2_TOTAL_TONNES")
        .select([
            pl.col("FLTNO"), 
            pl.col("DEPCITY"),
            pl.col("ARRCITY"),
            
            # pl.col("DEPAPT"),
            # pl.col("ARRAPT"),
            
            pl.col("ARRTIM"),
            pl.col("DEPTIM"),
            pl.col("ELPTIM"),
            pl.col("ESTIMATED_CO2_TOTAL_TONNES"),
            pl.col("INTAPT")
        ])
    ).collect()
    
    print(result)

get_flights_score("LHR", "BOM", 2024, "01", "20")

print(len(set(emissions["DEPARTURE_AIRPORT"])))



def evaluate_naive_atOffice(
    inbound_map  # location: num coming from there
):
    # each office gets a score
    scores = {locationL: -1 for location in OFFICES}
    