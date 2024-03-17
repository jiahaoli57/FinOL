from finol.config import *
import matplotlib.pyplot as plt

import yfinance

import numpy as np
asset_code = 'AA'
start_date = '2023-01-01'
end_date = '2024-01-01'

df = yfinance.download(asset_code, start=start_date, end=end_date)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.reset_index()
df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']

# Create figure and axis objects
fig, ax1 = plt.subplots()

# Convert DATE column to numpy array
DATE_values = np.array(df['DATE'].astype(str))

# Convert other columns to numpy arrays
OPEN_values = np.array(df['OPEN'])
HIGH_values = np.array(df['HIGH'])
LOW_values = np.array(df['LOW'])
CLOSE_values = np.array(df['CLOSE'])
VOLUME_values = np.array(df['VOLUME'])

# Plot 'OPEN', 'HIGH', 'LOW', 'CLOSE' on the first axis
ax1.plot(DATE_values, OPEN_values, color='blue', label='OPEN')
ax1.plot(DATE_values, HIGH_values, color='green', label='HIGH')
ax1.plot(DATE_values, LOW_values, color='red', label='LOW')
ax1.plot(DATE_values, CLOSE_values, color='purple', label='CLOSE')

# Set x-axis tick marks and rotate labels
plt.xticks(np.arange(0, len(DATE_values), 50), rotation=45)

# Set left y-axis label and title
ax1.set_ylabel('Price')
ax1.set_title(asset_code)

# Create a second axis for VOLUME
ax2 = ax1.twinx()

# Plot 'VOLUME' on the second axis
ax2.plot(DATE_values, VOLUME_values, color='orange', label='VOLUME')

# Set right y-axis label
ax2.set_ylabel('Volume')

# Add legend
lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')

# Show the plot
plt.show()
