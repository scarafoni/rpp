"""
plug in a video and it evalutes it on the i3d network
TODO- make this work for flow too
"""

import sys
import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from skvideo.io import vread
from Research_Platform import video_helpers, pytorch_helpers, helpers
from Research_Platform.repos.pytorch_i3d import pytorch_i3d
import os

parser = helpers.processing_parser
parser.add_argument('vfile', type=str)

repodir = os.path.dirname(os.path.realpath(pytorch_i3d.__file__))


def get_classes():
    label_file = os.path.join(os.path.dirname(os.path.realpath(pytorch_i3d.__file__)), 'label_map.txt')
    with open(label_file) as f:
        lines = list(map(lambda x: x.strip(), f.readlines()))
    return lines


def main(vfile, mode='rgb'):
    """
    classifies an image, note that the vfilemust be 256x256
    :param vfile: file to read
    :param mode: rgb or flow
    :return: classification as a string and number
    """

    # setup the model
    if mode == 'flow':
        i3d = pytorch_i3d.InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(os.path.join(repodir, 'models/flow_imagenet.pt')))
    else:
        i3d = pytorch_i3d.InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load(os.path.join(repodir, 'models/rgb_imagenet.pt')))
    # i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    data = vread(vfile)
    data = (data/255.)
    data = cv2.normalize(data, None, -1, 1, norm_type=cv2.NORM_MINMAX)
    data = video_helpers.to_channels_first(data)
    inputs = torch.tensor(data)
    # wrap them in Variable
    inputs = Variable(inputs.cuda())[None, ...].float()
    t = inputs.size(2)

    per_frame_logits = i3d(inputs)
    # upsample to input size
    per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

    # compute classification loss (with max-pooling along time B x C x T)
    y_ = torch.max(per_frame_logits, dim=2)[0]
    # y_ = torch.mean(per_frame_logits, dim=2)
    y_ = int(pytorch_helpers.p2n(torch.argmax(y_, dim=1)).squeeze())

    classes = get_classes()
    c = classes[y_]
    print(f'predicted class- {c}')


if __name__ == '__main__':
    args = parser.parse_args()
    pytorch_helpers.torch_boiler_plate(args.gpu)
    v = args.vfile
    main(v)
