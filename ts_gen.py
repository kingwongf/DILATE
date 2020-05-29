import pandas as pd
import numpy as np

df = pd.DataFrame({"A":range(0,10,1),
                   "B":range(0,20,2),
                   "C":range(0,30,3)})

# print(df.values)
# print(np.roll(df.values, 3))

def ts_gen(df):
    df = df.values
    print(df.shape)
    batch_size= 3

def rolling_window(a, window, step_size):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
def roll_win(ndarr, window):
    '''

    :param ndarr: dim = (time, n_features)
    :param window:
    :return:
    '''
    print(np.array([ndarr[i:i+window] for i in range(ndarr.shape[0]-window+1)]))

print(df)
# print(rolling_window(df.values,2,2).shape)
# print(rolling_window(df.values,2,2))
roll_win(df.values, 4)