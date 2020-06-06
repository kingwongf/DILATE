import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size):
        super(EncoderRNN, self).__init__()  
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers

        ## TODO rerun to add dropout with two grulstm layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_grulstm_layers,batch_first=True, dropout=0.3)
        # self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)

    def forward(self, input, hidden): # input [batch_size, length T, dimensionality d]      
        output, hidden = self.gru(input, hidden)
        return output, hidden
    
    def init_hidden(self,device):
        #[num_layers*num_directions,batch,hidden_size]
        # print(f"batch_size in encoder init hidden state: {self.batch_size}")
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
        # return (torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device),
        #         torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device),
        #         torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device))

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers,fc_units, output_size):
        super(DecoderRNN, self).__init__()
        ##TODO trying to match input as batch size is axis=1, not first=> axis=0
        # self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_grulstm_layers, dropout=0.3)
        self.fc = nn.Linear(hidden_size, fc_units)
        self.out = nn.Linear(fc_units, output_size)         
        
    def forward(self, input, hidden):
        # print(f"decoder forward input size: {input.size()}")
        output, hidden = self.gru(input, hidden) 
        output = F.relu( self.fc(output) )
        output = self.out(output[-1])
        return output, hidden


class EncoderSimple(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers, batch_size):
        super(EncoderSimple, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers

        ## TODO rerun to add dropout with two grulstm layers
        self.simple = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64)
        )
        # self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)

    def forward(self, input, hidden):  # input [batch_size, length T, dimensionality d]
        output, hidden = self.simple(input)
        return output, hidden

    def init_hidden(self, device):
        # [num_layers*num_directions,batch,hidden_size]
        # print(f"batch_size in encoder init hidden state: {self.batch_size}")
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
        # return (torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device),
        #         torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device),
        #         torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device))


class DecoderSimple(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers, fc_units, output_size):
        super(DecoderSimple, self).__init__()
        ##TODO trying to match input as batch size is axis=1, not first=> axis=0
        # self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.simple = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64,output_size)
        )


    def forward(self, input, hidden):
        # print(f"decoder forward input size: {input.size()}")
        output, hidden = self.simple(input)
        return output, hidden

class simpleGRU(nn.Module):
    def __init__(self, encoder, decoder, target_length, device):
        super(simpleGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device

    def forward(self, x):
        input_length  = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:,ei:ei+1,:]  , encoder_hidden)
        # print(f"encoder output shape: {encoder_output.size()}")
        # rint(f"encoder_hidden shape: {encoder_hidden.size()}")
        # print(f"x, decoder_input size: {x.size()}")

        batch_size, num_steps, n_feats = x.size()
        decoder_input = torch.tensor([[0.0]] * batch_size, dtype=torch.float)
        decoder_input = decoder_input.unsqueeze(0)
        ## TODO trying to init decoder input with 0s
        # decoder_input = x[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence

        decoder_hidden = encoder_hidden

        outputs = torch.zeros([x.shape[0], self.target_length, 1] ).to(self.device)
        ## TODO changed last dim to 1 because trying to predict 1 feat only
        ## outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]  ).to(self.device)

        # print(f"outputs size: {outputs.size()}") ### orig seq2seq outputs size: torch.Size([49, 10, 1])
        for di in range(self.target_length):
            # print(f"decoder_input size: {decoder_input.size()}")

            # decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.unsqueeze(0)
            # print(f"successful di: {di}")
            # print(f"new decoder input size: {decoder_input.size()}")
            # print(f"new decoder output size: {decoder_output.size()}")
            # print(f"new decoder output size: {decoder_output.decoder_output.unsqueeze(-1).size()}")
            outputs[:,di:di+1,:] = decoder_output.unsqueeze(-1)
        return outputs

class Net_GRU(nn.Module):
    def __init__(self, encoder, decoder, target_length, device):
        super(Net_GRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device

    def forward(self, x):
        input_length  = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:,ei:ei+1,:]  , encoder_hidden)
        # print(f"encoder output shape: {encoder_output.size()}")
        # rint(f"encoder_hidden shape: {encoder_hidden.size()}")
        # print(f"x, decoder_input size: {x.size()}")

        batch_size, num_steps, n_feats = x.size()
        decoder_input = torch.tensor([[0.0]] * batch_size, dtype=torch.float)
        decoder_input = decoder_input.unsqueeze(0)
        ## TODO trying to init decoder input with 0s
        # decoder_input = x[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence

        decoder_hidden = encoder_hidden

        outputs = torch.zeros([x.shape[0], self.target_length, 1] ).to(self.device)
        ## TODO changed last dim to 1 because trying to predict 1 feat only
        ## outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]  ).to(self.device)

        # print(f"outputs size: {outputs.size()}") ### orig seq2seq outputs size: torch.Size([49, 10, 1])
        for di in range(self.target_length):
            # print(f"decoder_input size: {decoder_input.size()}")

            # decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.unsqueeze(0)
            # print(f"successful di: {di}")
            # print(f"new decoder input size: {decoder_input.size()}")
            # print(f"new decoder output size: {decoder_output.size()}")
            # print(f"new decoder output size: {decoder_output.decoder_output.unsqueeze(-1).size()}")
            outputs[:,di:di+1,:] = decoder_output.unsqueeze(-1)
        return outputs


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        print(f"orig input shape in lstm: {input_seq.shape}")
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length, target_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.target_length = target_length
        self.n_hidden = 20 # number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size = n_features,
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first = True)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True

        # self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
        self.l_linear = torch.nn.Linear(self.n_hidden, 1)


    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error

        # print(f"lstm out shape: {lstm_out.shape}")
        # print(f"target_length: {self.target_length}")
        x= lstm_out[:,-self.target_length:,:]
        # print(f"x shape : {x.shape}")
        # print(f"linear x shape: {self.l_linear(x).shape}")
        # print(f"linear x unsequeeze shape: {self.l_linear(x).unsqueeze(-1).shape}")
        return self.l_linear(x)