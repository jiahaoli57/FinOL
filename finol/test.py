from tabulate import tabulate

profitability_table = [
    ["Ranking", "NYSE(O)", "NYSE(N)", "DJIA", "SP500", "TSE", "SSE", "HSI", "CMEG", "CRYPTO"],
    ["1st", "ANTI2", "SSPO", "SSPO", "SSPO", "RPRT", "BCRP", "BCRP", "BCRP", "BCRP"],
    ["2nd", "ANTI1", "BCRP", "PPT", "PPT", "PPT", "Best", "Best", "Best", "Best"],
    ["3rd", "ONS", "ANTI2", "KTPT", "KTPT", "SSPO", "GRW", "SSPO", "SSPO", "ONS"],
    ["4th", "BCRP", "PPT", "PAMR", "ANTI1", "RMR", "ANTI2", "GRW", "PPT", "ANTI2"],
    ["5th", "PPT", "PAMR", "CWMR-Stdev", "ANTI2", "AICTR", "ANTI1", "CWMR-Stdev", "ANTI2", "GRW"]
]
# self.text_widget.insert(
#     tk.END,
#     "# Benchmark Results\n"
#     "## Profitability\n"
#     "As of July 2024, the top five baselines for profitability (quantified by the size of CW metric) are:\n"
# )
# self.text_widget.insert(
#     tk.END,
#     tabulate(profitability_table, headers="firstrow", tablefmt="psql", numalign="left")
# )
print(tabulate(profitability_table, headers="firstrow", tablefmt="psql", numalign="left"))