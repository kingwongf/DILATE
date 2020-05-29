import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, inputs):
        # Push through RNN layer (the ouput is irrelevant)
        _, self.hidden = self.lstm(inputs, self.hidden)
        return self.hidden


class Decoder(nn.Module):

    def __init__(self, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        # input_size=1 since the output are single values
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, outputs, hidden, criterion):
        batch_size, num_steps = outputs.shape
        # Create initial start value/token
        input = torch.tensor([[0.0]] * batch_size, dtype=torch.float)
        # Convert (batch_size, output_size) to (seq_len, batch_size, output_size)
        input = input.unsqueeze(0)

        loss = 0
        for i in range(num_steps):
            # Push current input through LSTM: (seq_len=1, batch_size, input_size=1)
            output, hidden = self.lstm(input, hidden)
            # Push the output of last step through linear layer; returns (batch_size, 1)
            output = self.out(output[-1])
            # Generate input for next step by adding seq_len dimension (see above)
            input = output.unsqueeze(0)
            # Compute loss between predicted value and true value
            loss += criterion(output, outputs[:, i])
        return loss


if __name__ == '__main__':

    # 5 is the number of features of your data points
    encoder = Encoder(5, 128)
    decoder = Decoder(128)
    # Create optimizers for encoder and decoder
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Some toy data: 2 sequences of length 10 with 5 features for each data point
    inputs = [
        [
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
        ],
        [
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
            [0.5, 0.2, 0.3, 0.4, 0.1],
        ]
    ]

    inputs = torch.tensor(np.array(inputs), dtype=torch.float)
    # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
    inputs = inputs.transpose(1,0)

    # 2 sequences (to match the batch size) of length 6 (for the 6h into the future)
    outputs = [ [0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.3, 0.2, 0.1] ]
    outputs = torch.tensor(np.array(outputs), dtype=torch.float)

    #
    # Do one complete forward & backward pass
    #
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Reset hidden state of encoder for current batch
    encoder.hidden = encoder.init_hidden(inputs.shape[1])
    # Do forward pass through encoder
    hidden = encoder(inputs)
    # Do forward pass through decoder (decoder gets hidden state from encoder)
    loss = decoder(outputs, hidden, criterion)
    # Backpropagation
    loss.backward()
    # Update parameters
    encoder_optimizer.step()
    decoder_optimizer.step()
    print("Loss:", loss.item())