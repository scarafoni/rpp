import torch
import torch.utils.data as data_utl
import numpy as np
import json
import os
import os.path
from Research_Platform import helpers


def make_dataset(config_file, split, rgb_root, num_classes):
    dataset = []
    with open(config_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['split'] != split:
            continue

        if not os.path.exists(os.path.join(rgb_root, vid)):
            print(f'ERROR, cannot find {os.path.join(rgb_root, vid)}')
            continue

        nframes = nframes_from_vid_folder(os.path.join(rgb_root, vid))
        window_size = window_size_from_fname(os.listdir(os.path.join(rgb_root, vid))[0])

        label = np.zeros((num_classes, nframes), np.float32)

        for ann in zip(data[vid]['actions'], data[vid]['start_frames'], data[vid]['end_frames']):
            for fr in range(ann[1], min(ann[2], nframes-1)):
                label[ann[0], fr] = 1  # binary classification

        ys = np.argmax(label, axis=0)
        per_window_labels = []
        for j in range(0, len(ys)-window_size):
            endj = min(len(ys), j+window_size)
            l = ys[j:endj]
            per_window_labels.append(helpers.most_common(l))

        assert(len(per_window_labels) == len(ys)-window_size)
        per_window_labels = np.array(per_window_labels)

        dataset.append((vid, label, data[vid]['duration'], nframes, per_window_labels))

        i += 1
    return dataset, window_size


def nframes_from_vid_folder(folder):
    maxx = 0
    for file in os.listdir(folder):
        basename = helpers.basename_without_extension(file)
        starti, endi = [int(x) for x in basename.split('_')[-1].split('-')]
        if endi > maxx:
            maxx = endi
    return maxx


def window_from_fname(fname):
    basename = helpers.basename_without_extension(fname)
    starti, endi = [int(x) for x in basename.split('_')[-1].split('-')]
    return starti, endi


def window_size_from_fname(fname):
    starti, endi = window_from_fname(fname)
    return endi - starti


def window_size_from_folder(folder):
    files = os.listdir(folder)
    ls = [window_size_from_fname(x) for x in files]
    return max(ls)


class I3D_Dataset(data_utl.Dataset):

    def __init__(self, config_file, split, rgb_root, nclasses, balance_classes, chunks_per_video=5):
        self.data, window_size = make_dataset(config_file, split, rgb_root, nclasses)
        self.config_file = config_file
        self.rgb_root = rgb_root
        self.current_class = 0
        self.nclasses = nclasses
        self.window_size = window_size
        self.balance_classes = balance_classes
        self.chunks_per_video = chunks_per_video

    def next_class(self):
        self.current_class = (self.current_class + 1) % self.nclasses

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        # for f in range (n
        vid, label, dur, nf, pfl = self.data[index]
        # get all start, end frame combinations (that exist, with a 50% overlap)
        rgb_features = []
        real_labels = []
        frames = []
        r = range(0, nf-self.window_size, self.chunks_per_video)
        # for i in range(0, nf-self.window_size, int((nf-self.window_size)/(self.chunks_per_video-1))):
        for i in r:
            start_f = i
            end_f = min(start_f + self.window_size, nf)

            x1 = os.path.exists(os.path.join(self.rgb_root, vid, f'i3d_features_{start_f}-{end_f}.npy'))
            if x1:
                rgb_feature = np.load(os.path.join(self.rgb_root, vid, f'i3d_features_{start_f}-{end_f}.npy'))

                rgb_features.append(rgb_feature)
                real_labels.append(label[:, start_f:end_f])
                frames.append([start_f,end_f])

        rgb_features = np.array(rgb_features)
        real_labels = np.array(real_labels)
        frames = np.array(frames)

        return torch.from_numpy(rgb_features), torch.from_numpy(real_labels), vid, frames

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # root = '/coc/pcba1/Datasets/public/mpii_cooking_2/videos_frames'
    root = '../NRI_Data/DA3D/i3d_features'
    dataset = I3D_Dataset('/srv/share/datasets/MSR-DailyActivities3D/config.json', 'train', root, 16, True)
    ls = []
    for i in range(len(dataset.data)):
        rbg, l, vid, frames = dataset[i]
        print(l.argmax(dim=0))
        ls.extend(l.data.numpy().argmax(axis=0).tolist())

