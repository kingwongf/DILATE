import numpy as np
import torch
import pandas as pd
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset, CustomDataset
from models.orig_seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from loss.dilate_loss import dilate_loss
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)

# parameters
batch_size = 49 ## TODO NEED TO CHNAGE FROM 100 TO 1 FOR CUSTOM TIME SERIES, BATCH SIZE ITERATE FROM N= 500
N = 98
## 40 time steps in each N time series
N_input = 30 ## first 20 time steps as input
N_output = 10  ## last 20 time steps to predict
sigma = 0.01
gamma = 0.01
alpha =0.5
'''

# Load synthetic dataset
X_train_input,X_train_target,X_test_input,X_test_target,train_bkp,test_bkp = create_synthetic_dataset(N,N_input,N_output,sigma)

# print(X_train_input)
print(X_train_input.shape, X_train_target.shape)
print(X_test_input.shape, X_test_target.shape)

dataset_train = SyntheticDataset(X_train_input,X_train_target, train_bkp)
dataset_test = SyntheticDataset(X_test_input,X_test_target, test_bkp)
trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1)

'''
## Load Custom time series

## limiting due to input and output step sizes and N size


def custom_train_test(df, N_input, N_output, split=0.5):
    arr = np.log(df.dropna(axis=0).values)

    ## input and target step sizes for both train and test sets
    # N_input = 30
    # N_output = 10

    N = len(arr) // (2 * (N_input + N_output))

    arr = arr[:N * (2 * (N_input + N_output))]

    arr_N = arr.reshape((N, int(len(arr) // N)))

    train, test = arr_N[:, :int(split * arr_N.shape[1])], arr_N[:, int(split * arr_N.shape[1]):]

    train_input, train_target = train[:, :N_input], train[:, N_input:]
    test_input, test_target = test[:, :N_input], test[:, N_input:]

    return train_input, train_target, test_input, test_target, N
'''
arr_us = pd.read_excel("data/data.xlsx", usecols=["US Equity"]).dropna(axis=0).values[:]
date_limit = len(arr_us)//(N_input+N_output)*(N_input+N_output)
arr_us = arr_us[:date_limit]
two_d = arr_us.reshape((197,40))
arr_train, arr_test = two_d[:,:20], two_d[:,20:]

train_us, test_us = arr_us[:int(len(arr_us)/2)].T, arr_us[int(len(arr_us)/2):].T

## Splitting first half as input and second half as target, like the synthetic dataset (20 steps input, 20 steps target)
input_train_us, target_train_us = train_us[:, :int(train_us.shape[1]/2)], train_us[:, int(train_us.shape[1]/2):]
input_test_us, target_test_us = test_us[:, :int(test_us.shape[1]/2)], test_us[:, int(test_us.shape[1]/2):]


print(arr_us.shape)
print(train_us.shape, test_us.shape)
print(input_train_us.shape, target_train_us.shape)
print(input_test_us.shape, target_test_us.shape)

## As N_input and N_output are reused in plotting the test set, we need set as the axis=1 dim
N_input = input_test_us.shape[1]
N_output = target_test_us.shape[1]

print(N_input, N_output)
'''
## US Equity
df = pd.read_excel("data/data.xlsx", usecols=["US Equity"])
train_input, train_target, test_input, test_target, N = custom_train_test(df, N_input=N_input,
                                                                       N_output=N_output, split=0.5)
print(N)

print(train_input.shape, train_target.shape, test_input.shape, test_target.shape)
dataset_train = CustomDataset(train_input, train_target)
dataset_test = CustomDataset(test_input, test_target)

trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1)



def train_model(net,loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50,eval_every=50, verbose=1, Lambda=1, alpha=0.5):
    
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
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
            batch_size, N_output = target.shape[0:2]                     

            # forward + backward + optimize
            outputs = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            
            if (loss_type=='mse'):
                loss_mse = criterion(target,outputs)
                loss = loss_mse                   
 
            if (loss_type=='dilate'):
                loss, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, gamma, device)
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          
        
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
        batch_size, N_output = target.shape[0:2]
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


encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
net_gru_dilate = Net_GRU(encoder,decoder, N_output, device).to(device)
train_model(net_gru_dilate,loss_type='dilate',learning_rate=0.001, epochs=500, gamma=gamma, print_every=50, eval_every=50,verbose=1)

encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
net_gru_mse = Net_GRU(encoder,decoder, N_output, device).to(device)
train_model(net_gru_mse,loss_type='mse',learning_rate=0.001, epochs=500, gamma=gamma, print_every=50, eval_every=50,verbose=1)

# Visualize results
gen_test = iter(testloader)
# test_inputs, test_targets, breaks = next(gen_test)
test_inputs, test_targets= next(gen_test)

test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
criterion = torch.nn.MSELoss()

nets = [net_gru_mse,net_gru_dilate]
# nets = [net_gru_dilate]

for ind in range(1,51):
    plt.figure()
    plt.rcParams['figure.figsize'] = (17.0,5.0)  
    k = 1
    for net in nets:
        pred = net(test_inputs).to(device)

        input = test_inputs.detach().cpu().numpy()[ind,:,:]
        target = test_targets.detach().cpu().numpy()[ind,:,:]
        preds = pred.detach().cpu().numpy()[ind,:,:]

        plt.subplot(1,2,k)
        plt.plot(range(0,N_input) ,input,label='input',linewidth=3)
        plt.plot(range(N_input-1,N_input+N_output), np.concatenate([ input[N_input-1:N_input], target ]) ,label='target',linewidth=3)   
        plt.plot(range(N_input-1,N_input+N_output),  np.concatenate([ input[N_input-1:N_input], preds ]) ,label='prediction',linewidth=3)
        plt.xticks(range(0,40,2))
        plt.legend()
        k = k+1

    plt.show()
