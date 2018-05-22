import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from LSTMCell import LSTMCell


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers,
                 dropout=0.65, idropout=0.4, rdropout=0.25,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.internal_drop = nn.Dropout(idropout)
        self.rdrop = nn.Dropout(rdropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.ModuleList([LSTMCell(ninp, nhid)] + [LSTMCell(nhid, nhid) for i in range(nlayers-1)])
        self.decoder = nn.Linear(ninp, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.predictor = nn.Sequential(nn.Linear(nhid, ninp),
                                       nn.BatchNorm1d(ninp),
                                       nn.Tanh())
        self.init_weights()
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.01
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden_states):
        ntimestep = input.size(0)
        bsz = input.size(1)
        emb = self.drop(self.encoder(input))
        hx, cx = hidden_states

        rmask = torch.ones(self.nlayers, self.nhid)
        if input.is_cuda:
            rmask = rmask.cuda()
        rmask = self.rdrop(rmask)

        output = []
        for i in range(input.size(0)):
            hi = emb[i]  # emb_i: bsz, nhid

            hy = []
            cy = []
            for j in range(self.nlayers):
                hyj, cyj = self.rnn[j](hi, (hx[j] * rmask[j], cx[j]))
                hy.append(hyj)
                cy.append(cyj)
                hi = self.internal_drop(hyj)

            output.append(hi)
            hx = torch.stack(hy, dim=0)
            cx = torch.stack(cy, dim=0)

        output = torch.stack(output, dim=0).view(-1, self.nhid)
        output = self.predictor(output)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded.view(ntimestep, bsz, decoded.size(1)), (hx, cx)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new(self.nlayers, bsz, self.nhid).zero_(), \
               weight.new(self.nlayers, bsz, self.nhid).zero_()
