import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
from Research_Platform import helpers

parser = argparse.ArgumentParser()
parser.add_argument('save_dir', type=str)
parser.add_argument('mode', type=str, help='rgb or flow')
parser.add_argument('load_model', type=str)
parser.add_argument('root', type=str)
parser.add_argument('vname', type=str)
parser.add_argument('nclasses', type=int)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--window-size', type=int, default=64)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
from i3d_dataset_full import I3D_Dataset as Dataset
from collections import OrderedDict


def convert_state_dict(d):
    # some state dicts have '.model' before variables, here we switch that
    newd = OrderedDict()

    for key, item in d.items():
        newd[key[7:]] = item

    return newd


def run(mode='rgb', root='/ssd2/charades/Charades_v1_rgb', split='charades/charades.json', batch_size=1, load_model='',
        save_dir='', vname='', nclasses=5, window_size=64):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'all', root, mode, nclasses, test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(args.nclasses)
    oldd = torch.load(load_model)
    newd = convert_state_dict(oldd)
    i3d.load_state_dict(newd)
    # i3d.load_state_dict(helpers.pickleload('ikea_state_dict.pkl'))
    i3d.cuda()
    i3d.eval() # Set model to evaluate mode
    vnames = [x[0] for x in dataset.data]
    idx = vnames.index(vname)
    inputs, labels, name = dataset[idx]
    inputs = inputs.view(1, *inputs.shape)

    b, c, t, h, w = inputs.shape
    eo = 0
    classifications = np.zeros([t, 4]) - 1
    for starti in range(0, int(t-window_size/4), int(window_size/4)):
        endi = int(min(starti + window_size, t))
        ip = Variable(torch.from_numpy(inputs.numpy()[:, :, starti:endi]).cuda(), volatile=True)
        r = i3d.forward(ip).squeeze(0).data.cpu().numpy()
        r = np.argmax(np.mean(r, axis=1))
        classifications[starti:endi, eo] = r

        eo = (eo + 1) % 4

    maxvote_classifications = np.zeros(t)
    for i in range(len(classifications)):
        classes = classifications[i, classifications[i] >= 0]
        maxvote_classifications[i] = helpers.most_common(classes)
    np.save(os.path.join(save_dir, name[0]), np.concatenate(maxvote_classifications, axis=0))


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, split=os.path.join(args.root, '../config.json'),
        load_model=args.load_model, save_dir=args.save_dir, vname=args.vname, nclasses=args.nclasses,
        window_size=args.window_size)
