import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from LSTMCell import LSTMCell


class RRNetModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers,
                 dropout=0.65, idropout=0.4, rdropout=0.25,
                 max_stack_size=2, tie_weights=False):
        super(RRNetModel, self).__init__()
        self.max_stack_size = max_stack_size
        # dropout
        self.drop = nn.Dropout(dropout)
        self.internal_drop = nn.Dropout(idropout)
        self.rdrop = nn.Dropout(rdropout)
        # rrnet cells
        self.rcell = LSTMCell(ninp, nhid)
        self.scell = LSTMCell(ninp, nhid)
        self.mcell = LSTMCell(ninp + nhid, nhid)
        # actor cell
        self.pi = nn.Sequential(nn.Linear(nhid + ninp, ninp),
                                nn.BatchNorm1d(ninp),
                                nn.ReLU(),
                                nn.Linear(ninp, 3))
        # encoder/decoder
        self.encoder = nn.Embedding(ntoken, ninp)
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

    def forward(self, input, hidden_states, stack=None, mode="sample"):
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
        seq_actions = []
        seq_logp_actions = []

        for i in range(input.size(0)):
            hi = emb[i]  # emb_i: bsz, nhid
            htm1, ctm1 = last_states
            # actions
            pi_cur = self.pi(torch.cat([htm1, hi], 1))
            mask = self._get_mask(stack, only_recur=(mode == "only_recur"))
            pi_cur.data.masked_fill_(mask, -float('inf'))
            pi_cur = F.softmax(pi_cur, 1)
            pi_cur_cat = torch.distributions.Categorical(pi_cur)
            actions_cur = pi_cur_cat.sample()
            actions_cur_logp = pi_cur_cat.log_prob(actions_cur)
            actions = actions_cur.squeeze().detach().data.cpu().numpy()
            # processes
            hr, cr = self.rcell(hi, (htm1, ctm1))
            hs, cs = self.scell(hi, (htm1, ctm1))
            si = self.emulate_stack_pop(stack)
            hm, cm = self.mcell(torch.cat([hi, si], -1), (htm1, ctm1))
            new_states = []
            # update hidden states and stack
            for b in range(B):
                # split
                assert len(stack[b]) <= self.max_stack_size
                if actions[b] == 0:
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
            last_states = (new_s, new_c)

            seq_actions.append(actions)
            seq_entropy.append((-pi_cur * torch.log(pi_cur + 1e-8)).sum(1))
            seq_logp_actions.append(actions_cur_logp)
            seq_outputs.append(new_s)

        self._vars = {
            'seq_entropy': torch.stack(seq_entropy, 0),
            'seq_actions': np.asarray(seq_actions),
            'seq_logp_actions': seq_logp_actions,
            'stack': self._detach_stack(stack)
        }

        seq_outputs = torch.stack(seq_outputs, dim=0).view(-1, self.nhid)
        seq_outputs = self.predictor(seq_outputs)
        seq_outputs = self.drop(seq_outputs)
        seq_decoded = self.decoder(seq_outputs)
        return seq_decoded.view(T, B, -1), (new_s, new_c)

    def compute_loss(self, outputs, targets):
        outputs = F.log_softmax(outputs, 2)
        loss = torch.gather(-outputs, 2, targets.unsqueeze(2)).squeeze()
        return loss.mean()

    def compute_rewards(self, outputs, targets, disc_gamma=0.98):
        R = 0.
        T, B = targets.size()
        # reward is 1 if most probable word index is the target, 0 sinon
        reward_matrix = (torch.max(outputs, 2)[1] == targets).float()
        # discounted reward matrix
        reward_matrix_disc = np.zeros((T, B))
        t, start = T - 1, T - 1
        while t >= 0:
            past_reward = (reward_matrix_disc[t + 1] if t != start else 0.)
            reward_matrix_disc[t] = disc_gamma * past_reward + reward_matrix[t]
            t -= 1
        reward_matrix_disc = torch.from_numpy(reward_matrix_disc).float().cuda()
        return reward_matrix_disc

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.nhid).zero_(), \
                weight.new(bsz, self.nhid).zero_())
