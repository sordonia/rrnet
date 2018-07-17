import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import Parameter
from torch.nn import init
from LSTMCell import LSTMCell


class RRNetModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers,
                 dropout=0.65, idropout=0.4, rdropout=0.25,
                 max_stack_size=2, tie_weights=False, only_recur=False):
        super(RRNetModel, self).__init__()
        self.max_stack_size = max_stack_size
        # dropout
        self.drop = nn.Dropout(dropout)
        self.internal_drop = nn.Dropout(idropout)
        self.rdrop = nn.Dropout(rdropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # standard lstm cells for lower layers
        self.lstms = [LSTMCell(ninp if nlayer == 0 else nhid, nhid)
                      for nlayer in range(nlayers - 1)]
        self.lstms = nn.ModuleList(self.lstms)
        # recurrent cell in the top layer, very simple implementation
        nbelow = ninp if nlayers == 1 else nhid
        self.rcell = LSTMCell(nbelow, nhid)
        # actor cell
        self.pi = nn.Sequential(nn.Linear(2 * nhid + nbelow, ninp),
                                nn.LayerNorm(ninp),
                                nn.Tanh(),
                                nn.Linear(ninp, 3))
        self.vf = nn.Sequential(nn.Linear(2 * nhid + nbelow, ninp),
                                nn.LayerNorm(ninp),
                                nn.Tanh(),
                                nn.Linear(ninp, 1))
        # encoder/decoder
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
        self._vars = {}
        self._sent = torch.zeros(self.nhid).cuda()

    def init_weights(self):
        initrange = 0.01
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_stack(self, bsz):
        return [[] for i in range(bsz)]

    def emulate_stack_pop(self, stack):
        return torch.stack([self._sent if not stack_ else stack_[-1]
                            for stack_ in stack], 0)

    def _detach_stack(self, stack):
        return [[h.detach() for h in s] for s in stack]

    def _get_mask(self, stack, only_recur=False):
        mask = []
        for s_ in stack:
            if only_recur:
                m_ = [1, 0, 1]
            else:
                m_ = [0, 0, 0]
                # cannot merge
                if len(s_) == 0:
                    m_[2] = 1
                # cannot split
                if len(s_) >= self.max_stack_size:
                    m_[0] = 1
            mask.append(m_)
        # assume it's on GPU
        return torch.cuda.ByteTensor(mask)

    def backward(self, input, stack=None):
        outputs, _ = self.backrnn(input)
        return outputs

    def forward(self, input, hidden_states, stack=None, only_recur=False, eps=0., argmax=False):
        T, B = input.size()
        if not stack:
            stack = self.init_stack(B)
        emb = self.drop(self.encoder(input))
        hx, cx = hidden_states

        rmask = torch.ones(self.nlayers, self.nhid)
        if input.is_cuda:
            rmask = rmask.cuda()
        rmask = self.rdrop(rmask)

        last_states = hidden_states

        seq_outputs = []
        seq_entropy = []
        seq_values = []
        seq_actions = []
        seq_logp_actions = []

        for i in range(input.size(0)):
            hi = emb[i]  # emb_i: bsz, nhid
            htm1, ctm1 = last_states

            # first evolve lstm
            next_states = [[], []]
            for l, lstm in enumerate(self.lstms):
                ht_l, ct_l = lstm(hi, (htm1[l] * rmask[l], ctm1[l]))
                next_states[0].append(ht_l)
                next_states[1].append(ct_l)
                hi = self.internal_drop(ht_l)

            htm1, ctm1 = (htm1[-1], ctm1[-1])

            # actions
            mask = self._get_mask(stack, only_recur=only_recur)
            si = self.emulate_stack_pop(stack)

            state = torch.cat([htm1, si, hi], 1)
            vi_cur = self.vf(state).squeeze()

            pi_cur = self.pi(state)
            pi_cur.data.masked_fill_(mask, -float('inf'))
            pi_cur = F.softmax(pi_cur, 1)
            pi_cur = (1. - mask.float()) * pi_cur
            pi_cur = pi_cur / pi_cur.sum(1, keepdim=True)

            # epsilon greedy exploration ?
            if eps > 0.:
                pi_rnd = pi_cur * 0.
                pi_rnd.data.masked_fill_(mask, -float('inf'))
                pi_rnd = F.softmax(pi_rnd, 1)
                pi_smp = (1. - eps) * pi_cur + eps * pi_rnd
            else:
                pi_smp = pi_cur

            if argmax:
                actions_cur = torch.max(pi_cur, 1)[1].unsqueeze(1)
            else:
                actions_cur = torch.multinomial(pi_smp, 1, replacement=True)
            actions_cur_logp = torch.log(torch.gather(pi_cur, 1, actions_cur).squeeze() + 1e-8)
            actions = actions_cur.squeeze().detach().data.cpu().numpy()

            # updates
            hr, cr = self.rcell(hi, (htm1 * rmask[-1], ctm1))
            # split is just the same, but the hr is saved in the stack
            hs, cs = hr, cr
            # merge is just a weighted average of the previous state
            hm, cm = (0.5 * hr + 0.5 * si), cr

            # update hidden states and stack
            new_states = []
            for b in range(B):
                # split
                if actions[b] == 0 and len(stack[b]) < self.max_stack_size:
                    new_states.append((hs[b], cs[b]))
                    stack[b].append(hs[b])
                elif actions[b] == 1:
                    new_states.append((hr[b], cr[b]))
                else:
                    # merge, pop from stack for real
                    assert len(stack[b]) > 0, "when merging, stack should be full"
                    new_states.append((hm[b], cm[b]))
                    stack[b] = stack[b][:-1]

            new_s = torch.stack([s[0] for s in new_states], 0)
            new_c = torch.stack([s[1] for s in new_states], 0)

            next_states[0].append(new_s)
            next_states[1].append(new_c)
            last_states = (torch.stack(next_states[0], 0),
                           torch.stack(next_states[1], 0))

            seq_values.append(vi_cur)
            seq_actions.append(actions)
            seq_entropy.append((-pi_cur * torch.log(pi_cur + 1e-8)).sum(1))
            seq_logp_actions.append(actions_cur_logp)
            seq_outputs.append(self.internal_drop(new_s))

        self._vars = {
            'seq_entropy': torch.stack(seq_entropy, 0),
            'seq_actions': np.asarray(seq_actions),
            'seq_values': torch.stack(seq_values, 0),
            'seq_logp_actions': torch.stack(seq_logp_actions, 0),
            'stack': self._detach_stack(stack)
        }

        seq_outputs = torch.stack(seq_outputs, dim=0).view(-1, self.nhid)
        seq_outputs = self.predictor(seq_outputs)
        seq_outputs = self.drop(seq_outputs)
        seq_decoded = self.decoder(seq_outputs)
        return seq_decoded.view(T, B, -1), last_states

    def compute_loss(self, outputs, targets):
        outputs = F.log_softmax(outputs, 2)
        loss = torch.gather(-outputs, 2, targets.unsqueeze(2)).squeeze()
        return loss.mean()

    def compute_pg_loss(self, outputs, targets, gamma=1.2):
        R = 0.
        T, B = targets.size()

        # reward is 1 if probability of the correct token is higher than 0.5
        p = F.softmax(outputs, 2)
        p = torch.gather(p, 2, targets.unsqueeze(2)).squeeze()
        reward_matrix = (p > 0.5).float()

        # discounted reward matrix
        reward_matrix_disc = np.zeros((T, B))
        t, start = T - 1, T - 1

        while t >= 0:
            past_reward = (reward_matrix_disc[t + 1] if t != start else 0.)
            reward_matrix_disc[t] = gamma * past_reward + reward_matrix[t]
            t -= 1

        reward_matrix_disc = torch.from_numpy(reward_matrix_disc).float().cuda()
        # baseline the rewards
        vf_loss = (self._vars['seq_values'] - reward_matrix_disc).pow(2.).mean()
        reward_matrix_disc -= self._vars['seq_values'].detach()
        rl_loss = torch.mean(-(self._vars['seq_logp_actions'] * reward_matrix_disc))
        return rl_loss, vf_loss

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.nlayers, bsz, self.nhid).zero_(), \
                weight.new(self.nlayers, bsz, self.nhid).zero_())
