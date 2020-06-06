import numpy as np
import torch
import pandas as pd
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset, CustomDataset, CustomDataset2d
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU, MV_LSTM
from sklearn.preprocessing import StandardScaler
from loss.dilate_loss import dilate_loss
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
from nn_soft_dtw import SoftDTW
import warnings
import warnings;



class process:
    def __init__(self):
        warnings.simplefilter('ignore')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        random.seed(0)

        ## US Equity
        df = pd.read_excel("data/data.xlsx", index_col='Date').ffill().dropna(axis=0)

        # parameters

        self.N = 98
        ## 40 time steps in each N time series
        self.N_input = 22  ## first 20 time steps as input
        self.N_output = 3  ## last 5 time steps to predict
        self.sigma = 0.01
        self.gamma = 1
        self.alpha = 0.5
        self.n_features = df.shape[1]
        self.seq_length = self.N_input
        self.target_col = "US Equity"

        self.shuffle_train = False


        self.hidden_size = 128 ## 64
        self.num_grulstm_layers = 3

        self.idx_tgt_col = df.columns.get_loc(self.target_col)

        self.df = df


    def train_test_roll_win(self,df, target_col, N_input, N_output, r, no_of_train):
        '''
        arr_length = total_no_batches * (1 + r) * (N_input + N_output) * no_of_train
        r: no. of train set/no. of test set
        no. of test = r * no. of train set, needs to be an intrger > 1

        :param ndarr: dim = (time, n_features)
        :param window:
        :return:
        '''
        # arr = np.array([ndarr[i:i+window] for i in range(ndarr.shape[0]-window+1)])
        ndarr = df.dropna(axis=0).values
        # print(((1+r)* (N_input + N_output) * no_of_train))
        total_no_batches = np.round(ndarr.shape[0] // ((1 + r) * (N_input + N_output) * no_of_train))

        resize_ndarr = ndarr[len(ndarr) - int(total_no_batches * (1 + r) * (N_input + N_output) * no_of_train):]
        print(f"total_no_batches: {total_no_batches}, ndarr: {ndarr.shape}, ndarr_resized: {resize_ndarr.shape}")
        print(f"N_input: {N_input}, N_output: {N_output}, r: {r}, no_of_train: {no_of_train}")

        # print(total_no_batches * (1 + r) * (N_input + N_output) * no_of_train)
        # print(int(total_no_batches * (1 + r) * (N_input + N_output) * no_of_train))
        assert len(resize_ndarr) == int(total_no_batches * (1 + r) * (N_input + N_output) * no_of_train)
        assert len(resize_ndarr) % total_no_batches == 0

        n_feat = resize_ndarr.shape[1]
        arr = resize_ndarr.reshape(int(total_no_batches),
                                   int(np.round((N_input + N_output) * (1 + r) * no_of_train)),
                                   n_feat)


        train_test_split = int((arr.shape[1] * 0.6))
        train, test = arr[:, :train_test_split, :], arr[:, train_test_split:, :]

        ## rolling window for train and test set
        window = N_input + N_output
        train = np.array(
            [train[i, j:j + window] for i in range(train.shape[0]) for j in range(train.shape[1] - window + 1)]
        )

        test = np.array([test[i, j:j + window] for i in range(test.shape[0])
                         for j in range(test.shape[1] - window + 1)])


        train_input, train_target = train[:, :N_input, :], train[:, -N_output:, self.idx_tgt_col]


        test_input, test_target = test[:, :N_input, :], test[:, -N_output:, self.idx_tgt_col]

        return train_input, train_target, test_input, test_target, int(total_no_batches)

    def train_test_roll_win_2(self,df, target_col, N_input, N_output, r):
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

        total_train_legnth = int((1 / r) * batch_size / (1 / r + 1))

        train_ndarr, test_ndarr = ndarr[:, :total_train_legnth], ndarr[:, total_train_legnth - N_input:]

        ## Scaling train and test for each segments
        seg_train_means = np.mean(train_ndarr, axis=1)
        seg_train_stds = np.std(train_ndarr, axis=1)

        print(seg_train_means.shape)
        print(test_ndarr.shape)

        ## Reshaping means and stds to test size
        train_means_test_size = np.broadcast_to(seg_train_means[:, np.newaxis, :], test_ndarr.shape)
        train_stds_test_size = np.broadcast_to(seg_train_stds[:, np.newaxis, :], test_ndarr.shape)
        # print(f"broadcast train scaling {np.broadcast_to(seg_train_means[:,np.newaxis,:], train_ndarr.shape).shape}")

        ## Scaling train and test sets with train means and stds
        train_ndarr = (train_ndarr - np.broadcast_to(seg_train_means[:, np.newaxis, :], train_ndarr.shape)) / \
                      np.broadcast_to(seg_train_stds[:, np.newaxis, :], train_ndarr.shape)

        test_ndarr = (test_ndarr - train_means_test_size) / train_stds_test_size

        print(f"train_means_test_size: {type(train_means_test_size)}")

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

        train_input, train_target = train_ndarr[:, :N_input, :], train_ndarr[:, -N_output:, idx_tgt_col]
        test_input, test_target = test_ndarr[:, :N_input, :], test_ndarr[:, -N_output:, idx_tgt_col]

        num_train, num_test = train_input.shape[0], test_input.shape[0]

        return train_input, train_target, test_input, test_target, test_means_ndarr, test_stds_ndarr, num_train, num_test




    def train_model(self,net, batch_size, loss_type, learning_rate, epochs=1000, gamma=0.001,
                    print_every=50, eval_every=50, verbose=1, Lambda=1, alpha=0.5):
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        criterion_softdtw = SoftDTW(gamma=gamma, normalize=True)

        for epoch in range(epochs):
            for i, data in enumerate(self.trainloader, 0):

                inputs, target = data

                inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
                target = torch.tensor(target, dtype=torch.float32).to(self.device)
                # batch_size, N_output = target.shape[0:2]

                # forward + backward + optimize

                outputs = net(inputs)
                loss_mse, loss_shape, loss_temporal = torch.tensor(0), torch.tensor(0), torch.tensor(0)

                ## TODO next run with dtw implementation
                if (loss_type == 'dtw'):
                    loss_dtw = criterion_softdtw(outputs, target)
                    loss = torch.mean(loss_dtw)
                if (loss_type == 'mse'):
                    loss_mse = criterion(target, outputs)
                    loss = loss_mse

                if (loss_type == 'dilate'):
                    loss, loss_shape, loss_temporal = dilate_loss(outputs, target, alpha, gamma, self.device)

                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (verbose):
                if (epoch % print_every == 0):
                    print('epoch ', epoch, ' loss ', loss.item(), ' loss shape ', loss_shape.item(), ' loss temporal ',
                          loss_temporal.item())
                    self.eval_model(net, self.testloader, batch_size, gamma, verbose=1,
                               target_mean=0, target_std=0)

    def eval_model(self, net, loader, batch_size, gamma, verbose=1,
                   target_mean=0, target_std=0):
        criterion = torch.nn.MSELoss()
        losses_mse = []
        losses_dtw = []
        losses_tdi = []

        for i, data in enumerate(loader, 0):
            loss_mse, loss_dtw, loss_tdi = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            # get the inputs
            inputs, target = data

            # inputs, target, breakpoints = data

            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            target = torch.tensor(target, dtype=torch.float32).to(self.device)
            # batch_size, N_output = target.shape[0:2]
            outputs = net(inputs)

            # MSE
            loss_mse = criterion(target, outputs)
            loss_dtw, loss_tdi = 0, 0
            # DTW and TDI
            for k in range(batch_size):
                target_k_cpu = target[k, :, 0:1].view(-1).detach().cpu().numpy()
                output_k_cpu = outputs[k, :, 0:1].view(-1).detach().cpu().numpy()

                loss_dtw += dtw(target_k_cpu, output_k_cpu)
                path, sim = dtw_path(target_k_cpu, output_k_cpu)

                Dist = 0
                for i, j in path:
                    Dist += (i - j) * (i - j)
                loss_tdi += Dist / (self.N_output * self.N_output)

            loss_dtw = loss_dtw / batch_size
            loss_tdi = loss_tdi / batch_size

            # print statistics
            losses_mse.append(loss_mse.item())
            losses_dtw.append(loss_dtw)
            losses_tdi.append(loss_tdi)
            ## TODO plotting eval


        print(' Eval mse= ', np.array(losses_mse).mean(), ' dtw= ', np.array(losses_dtw).mean(), ' tdi= ',
              np.array(losses_tdi).mean())

    def dataLoader(self, train_input, train_target, test_input, test_target):

        dataset_train = CustomDataset2d(train_input, train_target)
        dataset_test = CustomDataset2d(test_input, test_target)

        trainloader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=self.shuffle_train, num_workers=1)
        testloader = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=1)
        return trainloader, testloader

    def trainModelPipe(self, loss_type):
        encoder = EncoderRNN(input_size=self.n_features, hidden_size=self.hidden_size, num_grulstm_layers=self.num_grulstm_layers,
                             batch_size=self.batch_size).to(self.device)
        decoder = DecoderRNN(input_size=1, hidden_size=self.hidden_size, num_grulstm_layers=self.num_grulstm_layers,
                             fc_units=16, output_size=1).to(self.device)

        net_gru = Net_GRU(encoder, decoder, self.N_output, self.device).to(self.device)

        self.train_model(net_gru, batch_size=self.batch_size, loss_type=loss_type,
                    learning_rate=0.001, epochs=500, gamma=self.gamma, print_every=50,
                    eval_every=50, verbose=1, alpha=self.alpha
                    )

        return net_gru


    def test(self):
        gen_test = iter(self.testloader)
        test_inputs, test_targets = next(gen_test)

        test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(self.device)
        test_targets = torch.tensor(test_targets, dtype=torch.float32).to(self.device)
        criterion = torch.nn.MSELoss()

        return test_inputs, test_targets, criterion

    def testZeroOut(self, test_inputs, test_targets, criterion, net_gru_mse, net_gru_dtw, net_gru_dilate):
        ## Testing for first input/target loss

        criterion_softdtw = SoftDTW(gamma=self.gamma, normalize=True)

        zero_inputs = test_inputs.detach().cpu().numpy()[0, :, self.idx_tgt_col] * self.target_std + self.target_mean
        zero_targets = test_targets.detach().cpu().numpy()[0, :, :] * self.target_std + self.target_mean

        zero_mse_pred = net_gru_mse(test_inputs).to(self.device).detach().cpu().numpy()[0, :,
                        :] * self.target_std + self.target_mean
        zero_dtw_pred = net_gru_dtw(test_inputs).to(self.device).detach().cpu().numpy()[0, :,
                        :] * self.target_std + self.target_mean
        zero_dilate_pred = net_gru_dilate(test_inputs).to(self.device).detach().cpu().numpy()[0, :,
                           :] * self.target_std + self.target_mean

        print(f"zero input:{zero_inputs}")
        print(f"zero targets:{zero_targets}")
        print(f"zero mse:{zero_mse_pred}")
        print(f"zero dtw:{zero_dtw_pred}")
        print(f"zero dilate:{zero_dilate_pred}")
        print(f"mse net mse: {criterion(test_targets, net_gru_mse(test_inputs))}, "
              f"dtw: {criterion_softdtw(test_targets, net_gru_mse(test_inputs))}, "
              f"dilate: {dilate_loss(net_gru_mse(test_inputs), test_targets, alpha=self.alpha, gamma=self.gamma, device=self.device)} ")

        print(f"dtw net mse: {criterion(test_targets, net_gru_dtw(test_inputs))}, "
              f"dtw: {criterion_softdtw(test_targets, net_gru_dtw(test_inputs))}, "
              f"dilate: {dilate_loss(net_gru_dtw(test_inputs), test_targets, alpha=self.alpha, gamma=self.gamma, device=self.device)}, ")

        print(f"dilate net mse: {criterion(test_targets, net_gru_dilate(test_inputs))}, "
              f"dtw: {criterion_softdtw(test_targets, net_gru_dilate(test_inputs))}, "
              f"dilate: {dilate_loss(net_gru_dilate(test_inputs), test_targets, alpha=self.alpha, gamma=self.gamma, device=self.device)}, ")

    def main(self):
        # train_input, train_target, test_input, test_target, total_no_batches = self.train_test_roll_win(self.scaled_df, target_col=self.target_col, N_input=self.N_input, N_output=self.N_output,r=1 / 3, no_of_train=6)


        train_input, train_target, test_input, test_target, test_means_ndarr, test_stds_ndarr, num_train, num_test = \
            self.train_test_roll_win_2(self.df, target_col=self.target_col,
                                     N_input=self.N_input, N_output=self.N_output,
                                     r=1 / 3)

        print(f"train input: {train_input.shape}, "
              f"train target: {train_target.shape}, "
              f"test input: {test_input.shape}, "
              f"test target: {test_target.shape}")

        print(f"num_train:{num_train}, "
              f"num_test: {num_test}")

        ## can't call shape somehow
        # f"test_means_ndarr: {test_means_ndarr.shape}, "
        # f"test_stds_ndarr: {test_stds_ndarr.shape}, "



        ## TODO change back after fixing pred
        ## TODO output size from preds: (1, 1) to match target: (5, 1)

        # batch_size = int(total_no_batches/13)
        # self.batch_size = batch_size = int(min(num_train//23, num_test//23))
        self.batch_size = batch_size = 1


        self.trainloader, self.testloader = self.dataLoader(train_input, train_target, test_input, test_target)

        print(f"batch_size: {self.batch_size}")

        print(f"testLoader length: {len(self.testloader)}")

        net_gru_dtw = self.trainModelPipe(loss_type='dtw')
        net_gru_mse = self.trainModelPipe(loss_type='mse')
        net_gru_dilate = self.trainModelPipe(loss_type='dilate')

        nets = [net_gru_mse, net_gru_dtw, net_gru_dilate]
        nets_name = ["net_gru_mse", "net_gru_dtw", "net_gru_dilate"]

        # test_inputs, test_targets, criterion = self.test()


        # test_inputs, test_targets = next(gen_test)


        # print(f"torch test_inputs size:{test_inputs.size()}")

        gen_test = iter(self.testloader)
        losses_mse = []
        losses_dtw = []
        losses_tdi = []

        for ind, (test_inputs, test_targets) in enumerate(gen_test):
            test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(self.device)
            test_targets = torch.tensor(test_targets, dtype=torch.float32).to(self.device)
            criterion = torch.nn.MSELoss()
            fig, axs = plt.subplots(1, 3, sharey='col')
            for i, net in enumerate(nets):
                test_preds = net(test_inputs).to(self.device)

                loss_mse = criterion(test_targets, test_preds)

                # DTW and TDI
                loss_dtw, loss_tdi = 0, 0
                for k in range(batch_size):
                    target_k_cpu = test_targets[k, :, 0:1].view(-1).detach().cpu().numpy()
                    output_k_cpu = test_preds[k, :, 0:1].view(-1).detach().cpu().numpy()

                    loss_dtw += dtw(target_k_cpu, output_k_cpu)
                    path, sim = dtw_path(target_k_cpu, output_k_cpu)

                    Dist = 0
                    for i, j in path:
                        Dist += (i - j) * (i - j)
                    loss_tdi += Dist / (self.N_output * self.N_output)

                loss_dtw = loss_dtw / batch_size
                loss_tdi = loss_tdi / batch_size

                # print statistics
                losses_mse.append(loss_mse.item())
                losses_dtw.append(loss_dtw)
                losses_tdi.append(loss_tdi)

                input = test_inputs.detach().cpu().numpy()[ind, :, :]
                target = test_targets.detach().cpu().numpy()[ind, :, :]
                preds = test_preds.detach().cpu().numpy()[ind, :, :]

                ## select target column in input
                input = input[:, self.idx_tgt_col]

                ## Scaling back to original
                input = input * test_stds_ndarr[ind] + test_means_ndarr[ind]
                target = target * test_stds_ndarr[ind] + test_means_ndarr[ind]
                preds = preds * test_stds_ndarr[ind] + test_means_ndarr[ind]

                axs[i].plot(range(0, self.N_input), input, label='input', linewidth=1)

                axs[i].plot(range(self.N_input - 1, self.N_input + self.N_output),
                            np.concatenate([input[self.N_input - 1:self.N_input],
                                            target.ravel()]),
                            label='target', linewidth=1)

                axs[i].plot(range(self.N_input - 1, self.N_input + self.N_output),
                            np.concatenate([input[self.N_input - 1:self.N_input],
                                            preds.ravel()]),
                            label='prediction', linewidth=1)
                # axs[i].xticks(range(0,40,2))
                axs[i].legend()
                axs[i].set_title(f"{nets_name[i]} MSE: {round(loss_mse.item(),3)}, DTW: {round(loss_dtw,3)}. TDI: {round(loss_tdi,3)}")

            # plt.show()
            plt.savefig(f"results/new_test_loader/{ind}.png")

        print(' Test mse= ', np.array(losses_mse).mean(), ' dtw= ', np.array(losses_dtw).mean(), ' tdi= ',
              np.array(losses_tdi).mean())


        exit()


        for ind in range(1, 39):
            fig, axs = plt.subplots(1, 3, sharey='col')
            for i, net in enumerate(nets):
                pred = net(test_inputs).to(self.device)



                input = test_inputs.detach().cpu().numpy()[ind, :, :]
                target = test_targets.detach().cpu().numpy()[ind, :, :]
                preds = pred.detach().cpu().numpy()[ind, :, :]


                ## select target column in input
                input = input[:, self.idx_tgt_col]

                ## Scaling back to original
                input = input * self.target_std + self.target_mean
                target = target * self.target_std + self.target_mean
                preds = preds * self.target_std + self.target_mean

                print("before exponential")
                print(f"input: {input}, target: {target}, preds: {preds}")

                # input, target, preds = np.e**input, np.e**target, np.e**preds
                print("after exponential")
                print(f"input: {input}, target: {target}, preds: {preds}")

                print(f"input: {input.shape}, target: {target.shape}, preds: {preds.shape}")

                axs[i].plot(range(0, self.N_input), input, label='input', linewidth=1)

                axs[i].plot(range(self.N_input - 1, self.N_input + self.N_output),
                            np.concatenate([input[self.N_input - 1:self.N_input],
                                            target.ravel()]),
                            label='target', linewidth=1)

                axs[i].plot(range(self.N_input - 1, self.N_input + self.N_output),
                            np.concatenate([input[self.N_input - 1:self.N_input],
                                            preds.ravel()]),
                            label='prediction', linewidth=1)
                # axs[i].xticks(range(0,40,2))
                axs[i].legend()
                axs[i].set_title(nets_name[i])

            plt.show()




m = process()
m.main()

'''
for z in range(len(outputs.to(device).detach().cpu().numpy())):
    preds_arr = outputs.to(device).detach().cpu().numpy()[z, :, :] * target_std + target_mean
    input_arr = inputs.detach().cpu().numpy()[z, :, idx_tgt_col] * target_std + target_mean
    target_arr = target.detach().cpu().numpy()[z, :, :] * target_std + target_mean

    plt.plot(range(0, len(input_arr)), input_arr, label='input', linewidth=1)

    plt.plot(range(len(input_arr) - 1, len(input_arr) + len(target_arr)),
             np.concatenate([input_arr[len(input_arr) - 1:len(input_arr)],
                             target_arr.ravel()]),
             label='target', linewidth=1)

    plt.plot(range(len(input_arr) - 1, len(input_arr) + len(target_arr)),
             np.concatenate([input_arr[len(input_arr) - 1:len(input_arr)],
                             preds_arr.ravel()]),
             label='prediction', linewidth=1)
    plt.title(
        f"f{losses_mse}: {np.array(losses_mse).mean()}, "
        f"loss dtw: {np.array(losses_dtw).mean()}, "
        f"loss dti: {np.array(losses_tdi).mean()}")
    plt.show()
'''

'''
## TODO run with dtw implementation
encoder = EncoderRNN(input_size=self.n_features, hidden_size=128, num_grulstm_layers=2, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=2, fc_units=16, output_size=1).to(device)
net_gru_dtw = Net_GRU(encoder, decoder, N_output, device).to(device)

train_model(net_gru_dtw, batch_size=batch_size, loss_type='dtw',
            learning_rate=0.001, epochs=500, gamma=gamma, print_every=50,
            eval_every=50, verbose=1, alpha=alpha, target_mean=target_log_mean, target_std=target_log_std)

encoder = EncoderRNN(input_size=n_features, hidden_size=128, num_grulstm_layers=2, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=2, fc_units=16, output_size=1).to(device)
net_gru_dilate = Net_GRU(encoder, decoder, N_output, device).to(device)

# net_gru_dilate = MV_LSTM(n_features,seq_length, N_output).to(device)
train_model(net_gru_dilate, batch_size=batch_size, loss_type='dilate',
            learning_rate=0.001, epochs=500, gamma=gamma, print_every=50,
            eval_every=50, verbose=1, alpha=alpha, target_mean=target_log_mean, target_std=target_log_std)

encoder = EncoderRNN(input_size=n_features, hidden_size=128, num_grulstm_layers=2, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=2, fc_units=16, output_size=1).to(device)
net_gru_mse = Net_GRU(encoder, decoder, N_output, device).to(device)
# net_gru_mse = MV_LSTM(n_features,seq_length, N_output).to(device)

train_model(net_gru_mse, batch_size=batch_size, loss_type='mse',
            learning_rate=0.001, epochs=500, gamma=gamma, print_every=50,
            eval_every=50, verbose=1, alpha=alpha, target_mean=target_log_mean, target_std=target_log_std)
'''

'''
# Visualize results
gen_test = iter(testloader)
test_inputs, test_targets = next(gen_test)

test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
criterion = torch.nn.MSELoss()
'''



######################################################################################################################
############################################## RAN ABOVE ON CONSOLE ##################################################
######################################################################################################################



'''

pred_net_gru_mse = net_gru_mse(test_inputs).to(device)
print(pred_net_gru_mse.detach().cpu().numpy())
pred_net_gru_mse = pred_net_gru_mse.detach().cpu().numpy()


pred_net_gru_dilate = net_gru_dilate(test_inputs).to(device)
pred_net_gru_dilate = pred_net_gru_dilate.detach().cpu().numpy()

pred_net_gru_dtw = net_gru_dtw(test_inputs).to(device)
pred_net_gru_dtw = pred_net_gru_dtw.detach().cpu().numpy()


input = test_inputs.detach().cpu().numpy()
print(input)


target =  test_targets.detach().cpu().numpy()
print(target)
'''

######################################################################################################################



