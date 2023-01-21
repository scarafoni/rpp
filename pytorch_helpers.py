"""
pytorch helper functions
"""
import os
from Research_Platform import helpers
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import random
import numpy as np

def accuracy(y, y_, w=None):
    if len(y_.size()) > 1:
        y_ = torch.argmax(y_, 1)
    if len(y_.size()) > 1:
        y = torch.argmax(y, 1)
    if w is not None:
        return torch.sum(torch.mul((y == y_).float(), w)).float()/float(torch.sum(w))
    else:
        return torch.sum(y == y_)/float(len(y))


def reset_random_elements(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def torch_boiler_plate(gpu=0):
    if type(gpu) == int:
        gpu = [int(gpu)]
    else:
        gpu = [int(x) for x in gpu]

    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        gpu = [devices[x] for x in gpu]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu)
    # print('$$$$$$$$$$$$', os.environ['CUDA_VISIBLE_DEVICES'])
    # torch.cuda.set_device(0)


def get_trainable_parameters_pytorch(net):
    # get sthe trainable parameters for a pytorch network
    trainable = []
    for name, param in net.named_parameters():
        if param.requires_grad:
            trainable.append([name, param])
    return trainable


def print_trainable_parameters_pytorch(net):
    # get sthe trainable parameters for a pytorch network
    for name, param in net.named_parameters():
        print(name, param.requires_grad)


def onehot_to_ind_pytorch(y):
    return torch.argmax(y, dim=1)


def p2n(x):
    return x.detach().cpu().data.numpy()

def p2l(x):
    return x.cpu().data.numpy().tolist()

def save_torch_statedict(net, path):
    torch.save(net.state_dict(), path)


def load_torch_statedict(model, path):
    model.load_state_dict(torch.load(path))


def load_ptl_sytle_statedict(model, path):
    """ when you train a model in pytorch lightningn and save it the state dict has different keys. if you try to load
    the lightning script  as a standard model, it can give errors. Here we make it so that you can load it properly"""

    weights = torch.load(path)['state_dict']
    newdict = OrderedDict()
    for key, value in weights.items():
        newkey = '.'.join(key.split('.')[1:])
        newdict[newkey] = value
    model.load_state_dict(newdict)


def eq(t1, t2):
    return torch.allclose(t1, t2, 5)


class ArrayDataset(Dataset):
    "Sample numpy array dataset"

    def __init__(self, x, y, c=None):
        self.x = x
        self.y = y
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


def dlnext(d):
    return next(iter(d))


def is_net_cuda(net):
    return next(net.parameters()).is_cuda


def feature_extractable(net):
    """
    makes the model so that it's a feature extraction model (i.e. cuts off the top)6
    :param model: pytorch model
    :return:
    """

    modules = list(net.children())[:-1]
    model = nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False

    return model


def sort_by_length(lengths, *args):
    # sorts all the pytorch lists in *args by lengths (good for RNNs)
    lengths, ii = lengths.sort(descending=True)
    newargs = [lengths]
    for arg in args:
        newargs.append(arg[ii])

    return newargs


def softmax(x):
    return F.softmax(x, dim=1)


def standard_dataloader(dataset, pin_memory=False, num_workers=4, shuffle=True, batch_size=128, collate_fn=None):
    if collate_fn is not None:
        return DataLoader(dataset, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)
    else:
        return DataLoader(dataset, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle, batch_size=batch_size)


def freeze_layer(module):
    for p in module.parameters():
        p.requires_grad = False

if __name__ == '__main__':
    lengths = torch.randn(100)*100
    x = torch.randn(100, 75, 512)
    y = torch.randint(0, 18, (100,))
    lengths, x, y = sort_by_length(lengths, x, y)


def make_mask(ys, lens):
    """
    creates a maks of the same shape as the y's so that anything over whatever the length is is 0 and everything else is 1
    :param ys: labels
    :param lens: lengths to use
    :return: mask of the same shape as ys
    """

    mask = torch.zeros_like(ys)
    # print('lens', lens[:, -1])
    for i in range(ys.shape[0]):
        mask[i, :lens[i]] = 1.

    return mask


def quickplot(x, name='test.png'):
    if type(x) != list:
        x = [x]
    x = [p2n(xx) for xx in x]
    helpers.quickplot(x, name)


def normalized_counter(data):
    data = p2n(data).reshape(-1)
    return helpers.normalized_counter(data)