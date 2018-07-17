import argparse
import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

import os
import data
import model_normal
import model_rrnet


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default=os.environ.get('PT_DATA_DIR', './data'),
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.003,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=6.485508340193558e-06,
                    help='weight decay')
parser.add_argument('--weight_eps', type=float, default=0.,
                    help='eps greedy')
parser.add_argument('--weight_entropy', type=float, default=0.02,
                    help='entropy reward')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=150,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--max_stack_size', type=int, default=2, metavar='N',
                    help='max stack size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to output layers (0 = no dropout)')
parser.add_argument('--idropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--rdropout', type=float, default=0.4,
                    help='dropout applied to recurrent states (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default=None,
                    help='path to save the final model')
parser.add_argument('--only_recur', action='store_true')
parser.add_argument('--load', type=str,  default=None,
                    help='path to save the final model')
parser.add_argument('--device', type=int, default=0,
                    help='select GPU')
parser.add_argument('--output_dir', type=str,
                    default=os.environ.get('PT_OUTPUT_DIR', 'model'))
args = parser.parse_args()

torch.cuda.set_device(args.device)
writer = SummaryWriter(args.output_dir)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(os.path.join(args.data, 'penn'))

def batchify(data, bsz, random_start_idx=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    if random_start_idx:
        start_idx = random.randint(0, data.size(0) % bsz - 1)
    else:
        start_idx = 0
    data = data.narrow(0, start_idx, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model_rrnet.RRNetModel(ntokens, args.emsize, args.nhid, args.nlayers,
                               args.dropout, args.idropout, args.rdropout,
                               max_stack_size=args.max_stack_size, tie_weights=args.tied)


if not (args.load is None):
    with open(args.load, 'rb') as f:
        model = torch.load(f)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return h.data
    else:
        if isinstance(h, list):
            return [repackage_hidden(v) for v in h]
        else:
            return tuple(repackage_hidden(v) for v in h)

def get_model_name(val_ppl=None, test_ppl=None):
    if val_ppl and test_ppl:
        return "model_val={:.2f}_test={:.2f}.pt".format(val_ppl, test_ppl)
    return "model.pt"


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    stack = None
    hidden = model.init_hidden(eval_batch_size)

    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden, stack=stack, argmax=True,
                               only_recur=args.only_recur)
        stack = model._vars['stack']
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).item()
        hidden = repackage_hidden(hidden)

    return total_loss / len(data_source)


nupdates = 0
eps_schedule = np.linspace(0., args.weight_eps, num=10)[::-1]


def repeat_batch(data, n_rep):
    T, B = data.size()
    data_ = data.transpose(0, 1).repeat(1, n_rep).reshape(B * n_rep, T).transpose(0, 1)
    return data_


def train(epoch):
    # Turn on training mode which enables dropout.
    model.train()
    global nupdates, eps_schedule

    lam_eps = eps_schedule[epoch] if epoch < len(eps_schedule) else 0.

    print("Training with lam_eps: {:.3f}".format(lam_eps))

    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    stack = None
    hidden = model.init_hidden(args.batch_size)
    train_data = batchify(corpus.train, args.batch_size, random_start_idx=True)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        targets = targets.view(data.size(0), data.size(1))
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        outputs, hidden = model(data, hidden, stack=stack, eps=lam_eps,
                                only_recur=args.only_recur)
        entropy = model._vars['seq_entropy']
        stack = model._vars['stack']

        ll_loss = model.compute_loss(outputs, targets)
        pg_loss, vf_loss = model.compute_pg_loss(outputs, targets, gamma=0.95)

        (ll_loss + pg_loss + 0.1 * vf_loss - float(args.weight_entropy) * entropy.mean()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        mean_entropy = entropy.mean().item()
        total_loss += ll_loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('epoch {:d}, {:d}/{:d} batches, lr {:02.6f}, ms/batch {:5.2f}, '
                  'loss {:.2f}, pg_loss {:.3f}, vf_loss {:.3f}, etpy {:.2f}, ppl {:.2f}'.format(
                      epoch, batch, len(train_data) // args.bptt, lr,
                      elapsed * 1000 / args.log_interval, cur_loss,
                      pg_loss.item(), vf_loss.item(),
                      mean_entropy, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    seq_actions = model._vars['seq_actions']
    print(", ".join([str(seq_actions[i][0]) for i in range(len(seq_actions))]))


# Loop over epochs.
lr = args.lr
best_val_loss = None
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0, 0.999), eps=1e-9, weight_decay=args.weight_decay)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, patience=5, threshold=0, min_lr=0.00005)
model_file = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch)
        val_loss = evaluate(val_data)
        test_loss = evaluate(test_data)
        print('epoch {:d}, valid loss {:5.2f}, valid ppl {:8.2f}'.format(
            epoch, val_loss, math.exp(val_loss)))
        val_ppl, test_ppl = math.exp(val_loss), math.exp(test_loss)
        writer.add_scalar('val_ppl', math.exp(val_loss), epoch)
        writer.add_scalar('tst_ppl', math.exp(test_loss), epoch)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            if model_file:
                os.remove(model_file)
            model_file = os.path.join(args.output_dir, get_model_name(val_ppl, test_ppl))
            print('Saving %s' % model_file)
            with open(model_file, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        scheduler.step(val_loss)

except KeyboardInterrupt:
    print('Exiting from training early')

# Load the best saved model.
with open(model_file, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('test loss {:5.2f}, test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
writer.close()
