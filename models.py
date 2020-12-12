import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import config

class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]
    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        #TODO remove this for models
        #v = v.unsqueeze(2)
        #v = v.unsqueeze(3)

        q = self.text(q, list(q_len.data))
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = apply_attention(v, a)
        combined = torch.cat([v, q], dim=1) #change this part?
        answer = self.classifier(combined)
        return answer


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop = 0.0):
        super(TextProcessor, self).__init__()

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

class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x

def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    #
    # if running
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled
