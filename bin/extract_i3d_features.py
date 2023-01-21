#!/usr/bin/env python

import sys
import os
sys.path.append('/coc/pcba1/dscarafoni3')
sys.path.append('/coc/pcba1/dscarafoni/Research_Platform/repos/pytorch_i3d')
sys.path.append(f'{os.environ["RP"]}/repos/pytorch_i3d')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, help='rgb or flow')
parser.add_argument('root', type=str)
parser.add_argument('config', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--load_model', type=str, default=os.path.join(os.environ['RP'], 'repos/pytorch_i3d/models/rgb_imagenet.pt'))
parser.add_argument('--log-file', type=str, default='log.txt')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--window-size', type=int, default=64)
parser.add_argument('--reset', action='store_true')
parser.add_argument('--avgpool', action='store_true')

args = parser.parse_args()

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
from i3d_dataset_full import I3D_Dataset as Dataset
import i3d_dataset_full
from Research_Platform import helpers, pytorch_helpers
import gc
pytorch_helpers.torch_boiler_plate(args.gpu)


def run(mode='rgb', root='/ssd2/charades/Charades_v1_rgb', split='charades/charades.json', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    if args.reset:
        print('resetting')
        helpers.reset_directory(save_dir)

    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    chunksize = 1000
    dataset = Dataset(split, 'all', root, mode, 101, test_transforms, chunksize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    helpers.maybe_create_dir(save_dir)

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(load_model))
    i3d.replace_logits(157)

    i3d.cuda()

    i3d.eval()
    # logging.basicConfig(filename=args.log_file,
    #                     filemode='a',
    #                     format='%(asctime)s %(levelname)s %(message)s',
    #                     level=logging.DEBUG
    # )

    with torch.no_grad():
        # Iterate over data.
        for data in dataloader:
            gc.collect()
            # get the inputs
            inputs, labels, name = data
            print(f'name- {name[0]}')
            if pytorch_helpers.p2n(labels[0, 0, 0]) != np.array(-1):
                print(f'extracting features from {name[0]} as a whole video')

                b,c,t,h,w = inputs.shape
                vid_dir = os.path.join(save_dir, name[0])
                helpers.maybe_create_dir(vid_dir)
                window_size = args.window_size
                for starti, endi in helpers.sliding_window_intervals(t, window_size, full_size_only=True):# range(0, t - window_size):
                    if os.path.exists(f'{vid_dir}/i3d_features_{starti}-{endi}.npy'):
                        # print(f'{vid_dir}/i3d_features_{starti}-{endi}.npy already exists, skipping')
                        continue
                    endi = int(min(starti + window_size, t))
                    ip = Variable(torch.from_numpy(inputs.numpy()[:, :, starti:endi]).cuda(), volatile=True)
                    m = i3d.extract_features(ip, avgpool=args.avgpool)
                    r = m.squeeze(0).data.cpu().numpy()
                    if torch.cuda.memory_allocated(0) > 110000000:
                        # print('resetting memory')
                        torch.cuda.empty_cache()
                    # print(f'{vid_dir}/i3d_features_{starti}-{endi}.npy')
                    np.save(os.path.join(vid_dir, f'i3d_features_{starti}-{endi}.npy'), r)
                    del ip, m, r
            else:
                name = name[0]
                print(f'video {name} has too many frames, breaking up now...')
                t = len(inputs)
                # b,c,t,h,w = inputs.shape
                vid_dir = os.path.join(save_dir, name)
                helpers.maybe_create_dir(vid_dir)
                window_size = args.window_size
                for startchunk, endchunk in helpers.sliding_window_intervals(t, chunksize, chunksize):
                    if startchunk > 0:
                        startchunk -= window_size
                    # print(f'chunks- {startchunk}-{endchunk}')
                    imgs = i3d_dataset_full.load_rgb_frames(dataset.root, name, startchunk, endchunk-startchunk)

                    imgs = dataset.transforms(imgs)
                    imgs = i3d_dataset_full.video_to_tensor(imgs)
                    imgs = imgs.view(1, *imgs.shape)
                    t2 = endchunk-startchunk # imgs.shape[2]
                    for starti, endi in helpers.sliding_window_intervals(t2, window_size, full_size_only=True):
                        # endi = int(min(starti + window_size, endchunk))
                        # print(f'{startchunk}.{starti}-{endchunk}.{endi}: {vid_dir}/i3d_features_{starti+startchunk}-{endi+startchunk}.npy')
                        if os.path.exists(os.path.join(vid_dir, f'i3d_features_{starti+startchunk}-{endi+startchunk}.npy')):
                            # print('path exists, continuing...')
                            continue

                        ip = Variable(torch.from_numpy(imgs.numpy()[:, :, starti:endi]).cuda(), volatile=True)
                        m = i3d.extract_features(ip, avgpool=args.avgpool)
                        r = m.squeeze(0).data.cpu().numpy()
                        if torch.cuda.memory_allocated(0):
                            torch.cuda.empty_cache()
                        np.save(os.path.join(vid_dir, f'i3d_features_{starti+startchunk}-{endi+startchunk}.npy'), r)
                        del ip, r, m
                    del imgs
    print("FINISHED")

if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir, split=args.config)
