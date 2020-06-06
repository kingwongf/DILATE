import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


def train_test_roll_win_2(df, target_col, N_input, N_output, r):
    '''
    arr_length = total_no_batches * (1 + r) * (N_input + N_output) * no_of_train
    r: no. of train set/no. of test set
    no. of test = r * no. of train set, needs to be an intrger > 1

    :param ndarr: dim = (time, n_features)
    :param window:
    :return:
    '''
    # arr = np.array([ndarr[i:i+window] for i in range(ndarr.shape[0]-window+1)])
    num_features = df.shape[1]
    idx_tgt_col = df.columns.get_loc(target_col)
    ndarr = df.dropna(axis=0).values

    batch_size = int((N_input + N_output) * (1 / r + 1))
    total_num_batches = int((ndarr.shape[0]) // batch_size)
    window = N_input + N_output

    ndarr = ndarr[-batch_size * total_num_batches:]
    ndarr = ndarr.reshape(total_num_batches, batch_size, num_features)

    print(f"ndarr shape:{ndarr.shape}")

    total_train_legnth = int((1 / r) * batch_size / (1 / r + 1))

    train_ndarr, test_ndarr = ndarr[:, :total_train_legnth], ndarr[:, total_train_legnth- N_input:]

    print(f"train ndarr shape {train_ndarr.shape}")
    print(f"test ndarr shape {test_ndarr.shape}")

    ## Scaling train and test for each segments
    seg_train_means = np.mean(train_ndarr, axis=1)
    seg_train_stds = np.std(train_ndarr, axis=1)
    print(f"train mean shape: {seg_train_means.shape}")
    print(f"train std shape: {seg_train_stds.shape}")

    print(f"test reshape mean size: {np.broadcast_to(seg_train_means[:,np.newaxis,:], test_ndarr.shape).shape}")

    ## Reshaping means and stds to test size
    train_means_test_size = np.broadcast_to(seg_train_means[:,np.newaxis,:], test_ndarr.shape)
    train_stds_test_size = np.broadcast_to(seg_train_stds[:,np.newaxis,:], test_ndarr.shape)
    # print(f"broadcast train scaling {np.broadcast_to(seg_train_means[:,np.newaxis,:], train_ndarr.shape).shape}")

    ## Scaling train and test sets with train means and stds
    train_ndarr = (train_ndarr - np.broadcast_to(seg_train_means[:,np.newaxis,:], train_ndarr.shape))/ \
                         np.broadcast_to(seg_train_stds[:,np.newaxis,:], train_ndarr.shape)

    test_ndarr = (test_ndarr - train_means_test_size)/ train_stds_test_size

    print(f"broadcast train shape: {train_ndarr.shape}")
    print(f"broadcast test shape: {test_ndarr.shape}")
    print(f"train_means_test_size: {train_means_test_size.shape}")
    print(f"train_stds_test_size: {train_stds_test_size.shape}")

    ## rolling window within train set
    train_ndarr = np.array(
        [train_ndarr[i, j:j + window] for i in range(train_ndarr.shape[0]) for j in
         range(train_ndarr.shape[1] - window + 1)]
    )
    test_ndarr = np.array(
        [test_ndarr[i, j:j + window] for i in range(test_ndarr.shape[0]) for j in
         range(test_ndarr.shape[1] - window + 1)]
    )
    ## reshaping mean and std for test set of rolling window
    test_means_ndarr = np.array(
        [train_means_test_size[i, j:j + window] for i in range(train_means_test_size.shape[0]) for j in
         range(train_means_test_size.shape[1] - window + 1)]
    )

    test_stds_ndarr = np.array(
        [train_stds_test_size[i, j:j + window] for i in range(train_stds_test_size.shape[0]) for j in
         range(train_stds_test_size.shape[1] - window + 1)]
    )

    print(f"train size: {train_ndarr.shape}")
    print(f"test size: {test_ndarr.shape}")
    print(f"test_means_ndarr size: {test_means_ndarr.shape}")
    print(f"test_stds_ndarr size: {test_stds_ndarr.shape}")

    # print(test_stds_ndarr[0, 2:2 + window])
    # print(np.transpose(test_ndarr, (0,2,1)))




    train_input, train_target = train_ndarr[:, :N_input, : ], train_ndarr[:, -N_output:, idx_tgt_col]
    test_input, test_target = test_ndarr[:, :N_input, : ], test_ndarr[:, -N_output:, idx_tgt_col]

    print(f"train input: {train_input.shape} train_target: {train_target.shape}")
    print(f"test input: {test_input.shape} test_target: {test_target.shape}")

    num_train, num_test = train_input.shape[0], test_input.shape[0]

    return train_input, train_target, test_input, test_target, test_means_ndarr, test_stds_ndarr, num_train, num_test




testingDF = pd.DataFrame({'A': range(0,1000), 'B':range(0, 2000,2), 'C':range(0, 3000,3), 'D':range(0, 4000,4)})


train_test_roll_win_2(testingDF, 'A', 20,5,r=1/3)
