import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop = 0.0):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0) #create lookup table to store embeddings of vocabulary
        #embedding_tokens = size of list of encoded questions
        #embedding_features = size of each question vector 
        #padding_idx = pads output with embedding vector

        self.drop = nn.Dropout(drop) #During training, randomly zeroes some of the elements of the input tensor with 
        #probability p using samples from a Bernoulli distribution. 
        #Each channel will be zeroed out independently on every forward call.

        self.tanh = nn.Tanh()

        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)

        self.features = lstm_features #size of hidden state and cell state at each time step

        self._init_lstm(self.lstm.weight_ih_l0) #weight_ih_l0 is learnable input-hidden weights for layer 0
        #shape = (4*hidden_size, input_size)

        self._init_lstm(self.lstm.weight_hh_l0) #weight_hh_l0 is learnable hidden-hidden weights for layer 0
        #shape = (4*hidden_size, hidden_size)

        self.lstm.bias_ih_l0.data.zero_() #(see above) -> shape = (4*hidden_size)
        self.lstm.bias_hh_l0.data.zero_() #(see above) -> shape = (4*hidden_size)
        
        #here we initialize weights
        init.xavier_uniform(self.embedding.weight) #embedding.weight gives the learnable weights of the module of 
                                                   #shape (num_embeddings, embedding_dim) initialized from N(0,1)
        #Fills the input Tensor with values according to the method described in Understanding the difficulty of 
        #training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform(w)


    def forward(self, q, q_len):
        embedded = self.embedding(q) #stores vector of indices corresponding to question
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)