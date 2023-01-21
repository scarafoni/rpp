import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Research_Platform.repos.pytorch_i3d import pytorch_i3d
from i3d_feats_dataset_whole_video import I3D_Dataset as Feature_Dataset
# from Research_Platform import I3D_Dataset as Feature_Dataset
from Research_Platform import helpers, pytorch_helpers
import mlflow
import os
from sklearn.metrics import accuracy_score
import numpy as np
from collections import defaultdict
import pandas as pd
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser = helpers.bp_parser
parser.add_argument('rgb_root', type=str)
parser.add_argument('config', type=str)
parser.add_argument('nclasses', type=int)
parser.add_argument('--window-size', type=int, default=64)
parser.add_argument('--chunks-per-video', type=int, default=5)
parser.add_argument('--per-chunk-labels', action='store_true')

__this_dir__ = os.path.dirname(os.path.abspath(__file__))


def get_datasets(config_file, rgb_root, nclasses, batch_size):
    test_dataset = Feature_Dataset(config_file, 'test', rgb_root, nclasses, balance_classes=False, chunks_per_video=args.chunks_per_video)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    dataloaders = {'test': test_dataloader}

    return dataloaders

def load_state_dict(f):
    w = torch.load(f)
    w2 = OrderedDict()
    for key, val in w.items():
        w2[key] = val

    return w2

def run(rgb_root='/ssd/Charades_v1_rgb',
        config_file='charades/charades.json',
        nclasses=158, batch_size=1):

    dataloaders = get_datasets(config_file, rgb_root, nclasses, batch_size)
    dataloader = dataloaders['test']

    # setup the model
    net = pytorch_i3d.InceptionI3d(400, in_channels=3)
    # net.load_state_dict(torch.load(os.path.join(__this_dir__, 'models/rgb_imagenet.pt')))
    net.replace_logits(nclasses)
    net.cuda()
    w = load_state_dict(os.path.join(args.logdir, 'best_model.pth'))
    # pytorch_helpers.load_torch_statedict(net, os.path.join(args.logdir, 'best_model.pth'))
    net.load_state_dict(w)
    net.cuda()
    net.eval()  # Set model to evaluate mode
    results = defaultdict(list)

    for data in dataloader:
        rgb_input, labels, vid, frames = data

        # wrap them in Variable
        rgb_input = Variable(rgb_input.cuda())[0]
        t = args.window_size
        labels = Variable(labels.cuda())[0]

        per_frame_logits = net(rgb_input, True)

        if args.per_chunk_labels:
            per_frame_logits = F.softmax(per_frame_logits, dim=1)
            per_frame_logits = torch.mean(per_frame_logits, dim=2)

            y_ = np.array(pytorch_helpers.p2n(torch.argmax(per_frame_logits, dim=1)))
            y = np.argmax(pytorch_helpers.p2n(labels), axis=1)
            y = np.mean(y, axis=1)
            y = np.reshape(y, [-1])

            # print(f'video- {vid[0]}, y = {y}, y_ = {y_}')
            results['name'].extend(vid*len(y))
            results['y'].extend(y)
            results['y_'].extend(y_)
            results['frames'].extend(pytorch_helpers.p2n(frames[0]).tolist())
        else:
            per_frame_logits = per_frame_logits.transpose(1, 2)
            per_frame_logits = per_frame_logits.contiguous().view(-1, per_frame_logits.shape[2])
            per_frame_logits = F.softmax(per_frame_logits, dim=1)

            per_frame_logits = torch.mean(per_frame_logits, dim=0)

            y_ = np.array(pytorch_helpers.p2n(torch.argmax(per_frame_logits)))
            y = np.argmax(pytorch_helpers.p2n(labels), axis=1)
            y = np.reshape(y, [-1])
            y = helpers.most_common(y)
            print(vid, y, y_)

            # print(f'video- {vid[0]}, y = {y}, y_ = {y_}')
            results['name'].append(vid[0])
            results['y'].append(y)
            results['y_'].append(y_)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.logdir, 'detailed_test_results.csv'))

    ys = np.array(results['y'])
    y_s = np.array(results['y_'])
    actions = ['drink', 'eat', 'read book', 'call cellphone', 'write on a paper', 'use laptop', 'use vacuum cleaner', 'cheer up',
               'sit still', 'toss paper', 'play game', 'lay down on sofa', 'walk', 'play guitar', 'stand up', 'sit down']
    fig, ax, cm = helpers.plot_confusion_matrix(np.array(ys), np.array(y_s), classes=np.arange(args.hierarchies_and_nclasses),
                                                normalize=True)
    fig.savefig(os.path.join(args.logdir, f'confusion_matrix_test_real.png'))
    mlflow.log_artifact(os.path.join(args.logdir, f'confusion_matrix_test_real.png'))
    np.save(os.path.join(args.logdir, f'confusion_matrix_test.npy'), cm)
    tot_accuracy = accuracy_score(ys, y_s)
    mlflow.log_metric('test accuracy final', tot_accuracy)


if __name__ == '__main__':
    # need to add argparse
    args = parser.parse_args()
    # mlflow.set_tracking_uri('/nethome/dscarafoni3/dev/NRI/mlruns')
    mlflow.set_experiment(args.logdir)
    pytorch_helpers.torch_boiler_plate(args.gpu)
    args.logdir = helpers.latest_experiment_dir(args.logdir, args.slurm, False)
    print('directory setup', args.logdir)

    # with mlflow.start_run(run_name=':'.join(args.logdir.split('/')[-2:])):
    r = helpers.pickleload(os.path.join(args.logdir, 'mlflowr.pkl'))
    with mlflow.start_run(run_id=r):
        run(rgb_root=args.rgb_root,
            config_file=args.config,
            nclasses=args.hierarchies_and_nclasses,
            batch_size=1)
