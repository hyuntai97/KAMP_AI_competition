import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, batch_size):
        super(LSTM, self).__init__()

        self.seq_len = seq_len 
        self.n_features = n_features # 8 or 4
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim
        self.batch_size = batch_size
        self.rep_dim = embedding_dim

        self.rnn1 = nn.LSTM(
          input_size=self.n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True,  # True = (batch_size, seq_len, n_features)
                            # False = (seq_len, batch_size, n_features) 
                            #default = false
          bias=False
        )
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=self.embedding_dim,
          num_layers=1,
          batch_first=True,
          bias=False
        )

    def forward(self, x):
        x = x.reshape((-1, self.seq_len, self.n_features)) 
        x, (_, _) = self.rnn1(x) 
        x, (hidden_n, _) = self.rnn2(x)

        return x, hidden_n.reshape((-1, 1, self.embedding_dim)) 

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, batch_size):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.hidden_dim = 2 * embedding_dim
        self.n_features = n_features # 8 or 4
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.rnn1 = nn.LSTM(
        input_size=self.embedding_dim,
        hidden_size=self.embedding_dim,
        num_layers=1,
        batch_first=True,
        bias=False
        )
        self.rnn2 = nn.LSTM(
        input_size=self.embedding_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True,
        bias=False
        
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features, bias=False)

    def forward(self, x):
        x = x.repeat(1, self.seq_len, 1) 

        # x = x.reshape((self.n_features, self.seq_len, self.embedding_dim)) 
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x) 
        # x = x.reshape((self.seq_len, self.hidden_dim)) 

        return self.output_layer(x) 

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, batch_size, device):
        super(LSTMAutoencoder, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = 2 * embedding_dim
        self.n_features = n_features # 8 or 4
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        device = device

        self.encoder = LSTM(seq_len, n_features, embedding_dim, batch_size).to(device)
        self.decoder = Decoder(seq_len, n_features, embedding_dim, batch_size).to(device)
    
    def forward(self, x):
        rep, h = self.encoder(x)

        x = self.decoder(h)
        return x