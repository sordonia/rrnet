import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import Parameter
from torch.nn import init
from LSTMCell import LSTMCell


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
            init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant_(self.sigma_weight, self.sigma_init)
            init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight,
                        self.bias + self.sigma_bias * self.epsilon_bias)

    def sample_noise(self):
        self.epsilon_weight.normal_()
        self.epsilon_bias.normal_()

    def remove_noise(self):
        self.epsilon_weight.zero_()
        self.epsilon_bias.zero_()


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
                                nn.LayerNorm(ninp),
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

    def sample_noise(self):
        def _helper(mod):
            for module in mod._modules.values():
                if isinstance(module, NoisyLinear):
                    module.sample_noise()
                else:
                    if len(module._modules):
                        _helper(module)
        _helper(self)

    def remove_noise(self):
        def _helper(mod):
            for module in mod._modules.values():
                if isinstance(module, NoisyLinear):
                    module.remove_noise()
                else:
                    if len(module._modules):
                        _helper(module)
        _helper(self)

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

    def forward(self, input, hidden_states, stack=None, mode="sample", eps=0.):
        if self.training:
            self.sample_noise()
        else:
            self.remove_noise()

        T, B = input.size()
        if not stack:
            stack = self.init_stack(B)
        emb = self.drop(self.encoder(input))
        hx, cx = hidden_states

        # TODO: use this
        rmask = torch.ones(self.nhid)
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
            mask = self._get_mask(stack, only_recur=(mode == "only_recur"))
            pi_cur = self.pi(torch.cat([htm1, hi], 1).detach())
            pi_cur.data.masked_fill_(mask, -float('inf'))
            pi_cur = F.softmax(pi_cur, 1)

            pi_rnd = pi_cur * 0.
            pi_rnd.data.masked_fill_(mask, -float('inf'))
            pi_rnd = F.softmax(pi_rnd, 1)

            # sampling
            pi_cur_cat = torch.distributions.Categorical(pi_cur)
            pi_smp_cat = pi_cur_cat
            if eps > 0.:
                pi_smp_cat = torch.distributions.Categorical(
                    (1. - eps) * pi_cur + eps * pi_rnd)
            actions_cur = pi_smp_cat.sample()
            actions_cur_logp = pi_cur_cat.log_prob(actions_cur)
            actions = actions_cur.squeeze().detach().data.cpu().numpy()

            # processes
            hr, cr = self.rcell(hi, (htm1 * rmask, ctm1))
            hs, cs = self.scell(hi, (htm1 * rmask, ctm1))
            si = self.emulate_stack_pop(stack)
            hm, cm = self.mcell(torch.cat([hi, si], -1), (htm1 * rmask, ctm1))
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
