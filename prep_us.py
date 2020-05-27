import pandas as pd
import numpy as np

arr_us = pd.read_excel("data/data.xlsx", usecols=["US Equity"]).dropna(axis=0).values

## input and target step sizes for both train and test sets
N_input =30
N_output = 10

N = len(arr_us)//(2*(N_input + N_output))

arr_us = arr_us[:N*(2*(N_input + N_output))]

arr_us_N = arr_us.reshape((N, int(len(arr_us)//N)))


split = 0.5

train, test = arr_us_N[:,:int(split*arr_us_N.shape[1])], arr_us_N[:,int(split*arr_us_N.shape[1]):]

train_input, train_target = train[:,:N_input], train[:,N_input:]
test_input, test_target = test[:,:N_input], test[:,N_input:]

print(arr_us.shape)
print(arr_us_N.shape)
print(train.shape)
print(test.shape)
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

