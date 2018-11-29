# Stock Predictor

**Version 1.0.0**

This is a simple recursive neural network that uses the previous 90 days of Samsung closing Stock Prices to predict the current day's closing price.

Requirements:
* Keras
* Matplotlib

---

## Data
We use Yahoo Stock Prices from 02/03/2015 to 21/02/2018 for training data. Our validation data set are Stock Prices from 02/03/2018 to 01/06/2018.
The data was collected from [Yahoo Finance](https://uk.finance.yahoo.com/quote/005930.KS/history?p=005930.KS).

---
## Improvements
By adding a feature of competitor Stock Prices e.g LG, we may increase the accuracy of the RNN.

