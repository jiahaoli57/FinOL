from tabulate import tabulate

profitability_table = [
    ["Ranking", "NYSE(O)", "NYSE(N)", "DJIA", "SP500", "TSE", "SSE", "HSI", "CMEG", "CRYPTO"],
    ["1st", "ANTI2", "SSPO", "SSPO", "SSPO", "RPRT", "BCRP", "BCRP", "BCRP", "BCRP"],
    ["2nd", "ANTI1", "BCRP", "PPT",  "PPT",  "PPT", "Best", "Best", "Best", "Best"],
    ["3rd", "ONS", "ANTI2", "KTPT", "KTPT", "SSPO", "GRW", "SSPO", "SSPO", "ONS"],
    ["4th", "BCRP", "PPT", "PAMR", "ANTI1", "RMR", "ANTI2", "GRW", "PPT", "ANTI2"],
    ["5th", "PPT", "PAMR", "CWMR-Stdev", "ANTI2", "AICTR", "ANTI1", "CWMR-Stdev", "ANTI2", "GRW"]
]
risk_resilience_table = [
    ["Ranking", "NYSE(O)", "NYSE(N)", "DJIA", "SP500", "TSE", "SSE", "HSI", "CMEG", "CRYPTO"],
    ["1st", "BCRP", "GRW", "KTPT", "Market", "ANTI2", "UP", "WAAS", "ONS", "ANTI2"],
    ["2nd", "Best", "EG", "PPT", "Best", "ANTI1", "UCRP", "Market", "GRW", "ANTI1"],
    ["3rd", "ANTI1", "WAAS", "SSPO", "UCRP", "RMR", "SP", "EG", "SP", "BCRP"],
    ["4th", "UCRP", "SP", "GRW", "BCRP", "OLMAR-S", "EG", "UCRP", "UCRP", "ONS"],
    ["5th", "SP", "UCRP", "Best", "UP", "PPT", "WAAS", "SP", "UP", "SP"]
]
practicality_table = [
    ["Ranking", "NYSE(O)", "NYSE(N)", "DJIA", "SP500", "TSE", "SSE", "HSI", "CMEG", "CRYPTO"],
    ["1st", "BCRP", "EG", "BCRP", "BCRP", "BCRP", "BCRP", "EG", "BCRP", "EG"],
    ["2nd", "EG", "UP", "EG", "EG", "EG", "EG", "UCRP", "EG", "UCRP"],
    ["3rd", "CW-OGD", "UCRP", "SP", "SP", "UCRP", "UCRP", "SP", "SP", "SP"],
    ["4th", "GRW", "SP", "UCRP", "UCRP", "SP", "SP", "WAAS", "UCRP", "WAAS"],
    ["5th", "UCRP", "WAAS", "WAAS", "UP", "WAAS", "WAAS", "BCRP", "WAAS", "BCRP"]
]
print("# Benchmark Results\n")
print("## Profitability\n")
print("As of July 2024, the top five baselines for profitability (quantified by the size of CW metric) are:\n")
print(tabulate(profitability_table, headers="firstrow", tablefmt="psql", numalign="left"))
print()

print("## Risk Resilience\n")
print("As of July 2024, the top five baselines for risk resilience (quantified by the size of MDD metric) are:\n")
print(tabulate(risk_resilience_table, headers="firstrow", tablefmt="psql", numalign="left"))
print()

print("## Practicality\n")
print("As of July 2024, the top five baselines for practicality (quantified by the size of ATO metric) are:\n")
print(tabulate(practicality_table, headers="firstrow", tablefmt="psql", numalign="left"))
print()