# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | Adjusted daily price and volume data.                            |


# For different data, We have some basic knowledge for them

## Daily price and volume data
$open: open price of the futures on that day.
$close: close price of the futures on that day.
$high: high price of the futures on that day.
$low: low price of the futures on that day.
$volume: volume of the futures on that day.
$factor: factor value of the futures on that day.
$open_interest: number of outstanding futures contracts that remain open.
$year, $month, $dayofweek, $dayofyear: timing information, for soybean futures exhibit some cyclicality.
$PMOM5, $PMOM10, $PMOM20: price momentum over 5, 10, and 20 days.
$VMOM5: volume momentum over 5 days.
$OIMOM5, $OIMOM20: open-interest momentum over 5 and 20 days.
$ROC60: 60-day price ratio (Ref($close,60)/$close).
$STD5: 5-day price volatility normalized by price.
$VSTD5: 5-day volume volatility normalized by volume.
$WVMA5, $WVMA60: weighted volatility of price–volume interaction over 5 and 60 days.
$RSQR5, $RSQR10, $RSQR20, $RSQR60: R-squared of rolling linear trend of close over 5, 10, 20, and 60 days.
$RESI5, $RESI10: normalized rolling regression residual over 5 and 10 days (trend deviation).
$CORR5, $CORR10, $CORR20, $CORR60: rolling correlation between price and log(volume+1) over 5, 10, 20, and 60 days.
$CORD5, $CORD10, $CORD60: rolling correlation between price ratio ($close/Ref($close,1)) and volume ratio (Log($volume/Ref($volume,1)+1)) over 5, 10, and 60 days.
$KLEN: intraday range ($high – $low) normalized by $open.
$KLOW: lower-tail structure: (min($open,$close) – $low) / $open.