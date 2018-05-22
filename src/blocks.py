import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def reshape(x, shape):
    x_reshape = x.contiguous().view(*shape)
    return x_reshape



def stick_breaking(logits):
    e = F.sigmoid(logits)
    z = (1 - e).cumprod(dim=1)
    p = torch.cat([e.narrow(1, 0, 1), e[:, 1:] * z[:, :-1]], dim=1)

    return p


def softmax(x, mask=None):
    max_x, _ = x.max(dim=-1, keepdim=True)
    e_x = torch.exp(x - max_x)
    if not (mask is None):
        e_x = e_x * mask
    out = e_x / (e_x.sum(dim=-1, keepdim=True) + 1e-8)

    return out


class AttentionDropout(nn.Module):
    def __init__(self, p=0.5):
        super(AttentionDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        if self.training and self.p > 0:
            p = input.data.new(input.size()).zero_() + (1 - self.p)
            mask = Variable(torch.bernoulli(p))
            output = input * mask  # bsz, nslots
            output = output / (output.sum(dim=1, keepdim=True) + 1e-8)
        else:
            output = input
        return output
