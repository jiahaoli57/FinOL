import akshare as ak
import mplfinance as mpf  # Please install mplfinance as follows: pip install mplfinance

stock_us_daily_df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
stock_us_daily_df = stock_us_daily_df.set_index(["date"])
stock_us_daily_df = stock_us_daily_df["2020-03-01": "2020-04-29"]
print(stock_us_daily_df)
stock_us_daily_df.columns = ["OPEN", 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
stock_us_daily_df.columns = [col.lower() for col in stock_us_daily_df.columns]
mpf.plot(stock_us_daily_df, type="candle", mav=(3, 6, 9), volume=True, show_nontrading=False)
mpf.plot(stock_us_daily_df, type="ohlc", mav=(3, 6, 9), volume=True, show_nontrading=False)
