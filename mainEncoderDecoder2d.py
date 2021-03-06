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
import warnings; warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)

# parameters

N = 98
## 40 time steps in each N time series
N_input = 20 ## first 20 time steps as input
N_output = 5  ## last 5 time steps to predict
sigma = 0.01
gamma = 1
alpha=1
n_features = 3
seq_length = N_input
target_col="US Equity"

## US Equity
df = pd.read_excel("data/data.xlsx", usecols=["US Equity", "US Bond", "UK Equity"])

## TODO Trying without log
## log_df = np.log(df)
log_df = df


scaled_log_df = (log_df - log_df.mean(axis=0))/log_df.std(axis=0)


## Load Custom time series

## limiting due to input and output step sizes and N size



def train_test_roll_win(df, target_col, N_input, N_output, r, no_of_train):
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
    total_no_batches = np.round(ndarr.shape[0]//((1+r)* (N_input + N_output) * no_of_train))

    resize_ndarr = ndarr[len(ndarr) - int(total_no_batches * (1+r) * (N_input + N_output) * no_of_train):]
    print(f"total_no_batches: {total_no_batches}, ndarr: {ndarr.shape}, ndarr_resized: {resize_ndarr.shape}")
    print(f"N_input: {N_input}, N_output: {N_output}, r: {r}, no_of_train: {no_of_train}")
    # print(total_no_batches * (1 + r) * (N_input + N_output) * no_of_train)
    # print(int(total_no_batches * (1 + r) * (N_input + N_output) * no_of_train))
    assert len(resize_ndarr) == int(total_no_batches * (1 + r) * (N_input + N_output) * no_of_train)
    assert len(resize_ndarr) % total_no_batches == 0

    idx_tgt_col = df.columns.get_loc(target_col)


    n_feat = resize_ndarr.shape[1]
    # arr = ndarr[ndarr.shape[0]-total_no_batches*2*(N_input+ N_output):]
    arr = resize_ndarr.reshape(int(total_no_batches), int(np.round((N_input+ N_output)* (1 + r) * no_of_train)),n_feat)

    # print(f"batch reshaped arr: {arr.shape}")



    train_test_split = int((arr.shape[1] * 0.6))
    # print(train_test_split)
    # print(arr.shape)
    train, test = arr[:, :train_test_split, :], arr[:, train_test_split:, :]





    ## rolling window for train and test set
    window = N_input + N_output
    train = np.array([train[i, j:j+window] for i in range(train.shape[0]) for j in range(train.shape[1]-window+1)])
    test = np.array([test[i, j:j+window] for i in range(test.shape[0]) for j in range(test.shape[1]-window+1)])

    # print(f"rolling train shape: {train.shape}")
    # print(f"rolling test shape: {test.shape}")
    # print(train)

    train_input, train_target = train[:, :N_input, :], train[:, -N_output:, idx_tgt_col]
     #print(f"train input, output shape: {train_input.shape, train_target.shape}")
    # print(train_input, train_target)

    test_input, test_target = test[:, :N_input, :], test[:, -N_output:, idx_tgt_col]
    # print(f"test input, output shape: {test_input.shape, test_target.shape}")

    return train_input, train_target, test_input, test_target, int(total_no_batches)

train_input, train_target, test_input, test_target, total_no_batches = \
    train_test_roll_win(scaled_log_df, target_col=target_col, N_input=N_input, N_output=N_output,r=1/3, no_of_train=6)

idx_tgt_col = df.columns.get_loc(target_col)

target_log_mean = log_df.mean(axis=0).iloc[idx_tgt_col]
target_log_std = log_df.std(axis=0).iloc[idx_tgt_col]

print(f"input mean: {log_df.mean(axis=0)}")
print(f"input std: {log_df.std(axis=0)}")
print(f"target_log_mean: {target_log_mean}, target_log_std: {target_log_std}")

## TODO change back after fixing pred
## TODO output size from preds: (1, 1) to match target: (5, 1)

# batch_size = int(total_no_batches/13)
batch_size = int(total_no_batches)


dataset_train = CustomDataset2d(train_input, train_target)
dataset_test = CustomDataset2d(test_input, test_target)

trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1)



def train_model(net, batch_size,loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50,eval_every=50, verbose=1, Lambda=1, alpha=0.5,
                target_mean=0, target_std=0):
    
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    criterion = torch.nn.MSELoss()
    criterion_softdtw = SoftDTW(gamma=gamma, normalize=True)
    
    for epoch in range(epochs): 
        for i, data in enumerate(trainloader, 0):

            ## TODO modified for CustomDS
            try:
                inputs, target, _ = data
            except:
                try:
                    inputs, target= data
                except:
                    pass


            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            # batch_size, N_output = target.shape[0:2]


            # forward + backward + optimize
            # print(f"input size: {inputs.size()}")
            # print(f"input size -1: {inputs.size(-1)}")
            # print(net)
            outputs = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)

            ## TODO next run with dtw implementation
            if(loss_type=='dtw'):
                loss_dtw = criterion_softdtw(outputs, target)
                loss = torch.mean(loss_dtw)
            if (loss_type=='mse'):
                loss_mse = criterion(target,outputs)
                loss = loss_mse                   
 
            if (loss_type=='dilate'):
                loss, loss_shape, loss_temporal = dilate_loss(outputs, target,alpha, gamma, device)

            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          
            if epoch > 300:
                for z in range(len(outputs.to(device).detach().cpu().numpy())):
                    preds_arr = outputs.to(device).detach().cpu().numpy()[z,:,:] * target_std + target_mean
                    input_arr = inputs.detach().cpu().numpy()[z,:,idx_tgt_col]* target_std+ target_mean
                    target_arr = target.detach().cpu().numpy()[z, :,:]* target_std+ target_mean

                    plt.plot(range(0, len(input_arr)), input_arr, label='input', linewidth=1)

                    plt.plot(range(len(input_arr) - 1, len(input_arr) + len(target_arr)),
                                np.concatenate([input_arr[len(input_arr)- 1:len(input_arr)],
                                                target_arr.ravel()]),
                                label='target', linewidth=1)

                    plt.plot(range(len(input_arr) - 1, len(input_arr) + len(target_arr)),
                                np.concatenate([input_arr[len(input_arr)- 1:len(input_arr)],
                                                preds_arr.ravel()]),
                                label='prediction', linewidth=1)
                    plt.title(f"f{loss_type}: {loss.item()}, loss shape: {loss_shape.item()}, loss temporal: {loss_temporal.item()}")
                    plt.show()





        if(verbose):
            if (epoch % print_every == 0):
                print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                eval_model(net,testloader, gamma,verbose=1)

  

def eval_model(net,loader, gamma,verbose=1):   
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []   

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # get the inputs

        ## TODO modified for CustomDS
        try:
            inputs, target, _ = data
        except:
            try:
                inputs, target = data
            except:
                pass

        # inputs, target, breakpoints = data

        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        # batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)
         
        # MSE    
        loss_mse = criterion(target,outputs)    
        loss_dtw, loss_tdi = 0,0
        # DTW and TDI
        for k in range(batch_size):         
            target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

            loss_dtw += dtw(target_k_cpu,output_k_cpu)
            path, sim = dtw_path(target_k_cpu, output_k_cpu)   
                       
            Dist = 0
            for i,j in path:
                    Dist += (i-j)*(i-j)
            loss_tdi += Dist / (N_output*N_output)            
                        
        loss_dtw = loss_dtw /batch_size
        loss_tdi = loss_tdi / batch_size

        # print statistics
        losses_mse.append( loss_mse.item() )
        losses_dtw.append( loss_dtw )
        losses_tdi.append( loss_tdi )

    print( ' Eval mse= ', np.array(losses_mse).mean() ,' dtw= ',np.array(losses_dtw).mean() ,' tdi= ', np.array(losses_tdi).mean()) 


print(f"batch_size: {batch_size}")

## TODO run with dtw implementation
encoder = EncoderRNN(input_size=3, hidden_size=128, num_grulstm_layers=2, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=2,fc_units=16, output_size=1).to(device)
net_gru_dtw = Net_GRU(encoder,decoder, N_output, device).to(device)

train_model(net_gru_dtw, batch_size =batch_size,loss_type='dtw',
            learning_rate=0.001, epochs=500, gamma=gamma, print_every=50,
            eval_every=50,verbose=1, alpha=alpha, target_mean=target_log_mean, target_std=target_log_std)

encoder = EncoderRNN(input_size=3, hidden_size=128, num_grulstm_layers=2, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=2,fc_units=16, output_size=1).to(device)
net_gru_dilate = Net_GRU(encoder,decoder, N_output, device).to(device)

# net_gru_dilate = MV_LSTM(n_features,seq_length, N_output).to(device)
train_model(net_gru_dilate, batch_size =batch_size,loss_type='dilate',
            learning_rate=0.001, epochs=500, gamma=gamma, print_every=50,
            eval_every=50,verbose=1, alpha=alpha, target_mean=target_log_mean, target_std=target_log_std)

encoder = EncoderRNN(input_size=3, hidden_size=128, num_grulstm_layers=2, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=2,fc_units=16, output_size=1).to(device)
net_gru_mse = Net_GRU(encoder,decoder, N_output, device).to(device)
# net_gru_mse = MV_LSTM(n_features,seq_length, N_output).to(device)

train_model(net_gru_mse, batch_size=batch_size,loss_type='mse',
            learning_rate=0.001, epochs=500, gamma=gamma, print_every=50,
            eval_every=50,verbose=1, alpha=alpha, target_mean=target_log_mean, target_std=target_log_std)



# Visualize results
gen_test = iter(testloader)
test_inputs, test_targets= next(gen_test)

test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
criterion = torch.nn.MSELoss()

nets = [net_gru_mse, net_gru_dtw,net_gru_dilate]
nets_name = ["net_gru_mse", "net_gru_dtw","net_gru_dilate"]

######################################################################################################################
############################################## RAN ABOVE ON CONSOLE ##################################################
######################################################################################################################

## Testing for first input/target loss

criterion_softdtw = SoftDTW(gamma=gamma, normalize=True)

zero_inputs = test_inputs.detach().cpu().numpy()[0,:,idx_tgt_col]* target_log_std+ target_log_mean
zero_targets = test_targets.detach().cpu().numpy()[0,:,:]* target_log_std+ target_log_mean
zero_mse_pred = net_gru_mse(test_inputs).to(device).detach().cpu().numpy()[0,:,:]* target_log_std+ target_log_mean
zero_dtw_pred = net_gru_dtw(test_inputs).to(device).detach().cpu().numpy()[0,:,:]* target_log_std+ target_log_mean
zero_dilate_pred = net_gru_dilate(test_inputs).to(device).detach().cpu().numpy()[0,:,:]* target_log_std+ target_log_mean

print(f"zero input:{zero_inputs}")
print(f"zero targets:{zero_targets}")
print(f"zero mse:{zero_mse_pred}")
print(f"zero dtw:{zero_dtw_pred}")
print(f"zero dilate:{zero_dilate_pred}")
print(f"mse net mse: {criterion(test_targets, net_gru_mse(test_inputs))}, "
      f"dtw: {criterion_softdtw(test_targets, net_gru_mse(test_inputs))}, "
      f"dilate: {dilate_loss(net_gru_mse(test_inputs), test_targets, alpha=alpha, gamma=gamma, device=device)} ")

print(f"dtw net mse: {criterion(test_targets, net_gru_dtw(test_inputs))}, "
      f"dtw: {criterion_softdtw(test_targets, net_gru_dtw(test_inputs))}, "
      f"dilate: {dilate_loss(net_gru_dtw(test_inputs), test_targets, alpha=alpha, gamma=gamma, device=device)}, ")

print(f"dilate net mse: {criterion(test_targets, net_gru_dilate(test_inputs))}, "
      f"dtw: {criterion_softdtw(test_targets, net_gru_dilate(test_inputs))}, "
      f"dilate: {dilate_loss(net_gru_dilate(test_inputs), test_targets, alpha=alpha, gamma=gamma, device=device)}, ")





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





for ind in range(1,39):
    fig, axs = plt.subplots(1, 3, sharey='col')
    for i, net in enumerate(nets):
        pred = net(test_inputs).to(device)

        input = test_inputs.detach().cpu().numpy()[ind,:,:]
        target = test_targets.detach().cpu().numpy()[ind,:,:]
        preds = pred.detach().cpu().numpy()[ind,:,:]

        ## select target column in input
        input = input[:,idx_tgt_col]



        ## Scaling back to original
        input = input * target_log_std+ target_log_mean
        target = target * target_log_std + target_log_mean
        preds = preds * target_log_std + target_log_mean

        print("before exponential")
        print(f"input: {input}, target: {target}, preds: {preds}")

        # input, target, preds = np.e**input, np.e**target, np.e**preds
        print("after exponential")
        print(f"input: {input}, target: {target}, preds: {preds}")




        print(f"input: {input.shape}, target: {target.shape}, preds: {preds.shape}")


        axs[i].plot(range(0,N_input) ,input,label='input',linewidth=1)

        axs[i].plot(range(N_input-1,N_input+N_output),
                    np.concatenate([ input[N_input-1:N_input],
                                     target.ravel() ]),
                    label='target',linewidth=1)

        axs[i].plot(range(N_input-1,N_input+N_output),
                    np.concatenate([ input[N_input-1:N_input],
                                     preds.ravel() ]),
                    label='prediction',linewidth=1)
        # axs[i].xticks(range(0,40,2))
        axs[i].legend()
        axs[i].set_title(nets_name[i])

    plt.show()

