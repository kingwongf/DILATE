## DILATE Loss Model for Financial Market Indices Forecasting
Using the purposed loss objective, DILATE here [paper](https://papers.nips.cc/paper/8672-shape-and-time-distortion-loss-for-training-deep-time-series-forecasting-models), a seq2seq model has been trained and outperformed out-of-sample in forecasting US Equity index. 


|     |   net_gru_mse |   net_gru_soft_dtw |   net_gru_dilate |
|-----|---------------|--------------------|------------------|
| MSE |         2.219 |              1.992 |           0.0364 |
| DTW |         2.239 |              1.921 |           0.0321 |
| TDI |         2.381 |              2.051 |           0.0236 |

```
@incollection{leguen19dilate,
title = {Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models},
author = {Le Guen, Vincent and Thome, Nicolas},
booktitle = {Advances in Neural Information Processing Systems},
pages = {4191--4203},
year = {2019}
}
```
