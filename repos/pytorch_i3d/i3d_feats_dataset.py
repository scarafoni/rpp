import torch
import torch.utils.data as data_utl
import numpy as np
import json
import os
import os.path
from Research_Platform import helpers
import cv2


def make_dataset(split_file, split, root, num_classes):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['split'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue

        num_files = len(os.listdir(os.path.join(root, vid)))
        nframes = nframes_from_vid_folder(os.path.join(root, vid))
        # print(os.path.join(root, vid))
        window_size = window_size_from_fname(os.listdir(os.path.join(root, vid))[0])

        label = np.zeros((num_classes, nframes), np.float32)

        for ann in zip(data[vid]['actions'], data[vid]['start_frames'], data[vid]['end_frames']):
            # print(ann, nframes)
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

        dataset.append((vid, label, nframes*30, nframes, per_window_labels))

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

    def __init__(self, split_file, split, root, nclasses, balance_classes):
        self.data, window_size = make_dataset(split_file, split, root, nclasses)
        self.split_file = split_file
        self.root = root
        self.current_class = 0
        self.nclasses = nclasses
        self.window_size = window_size
        self.balance_classes = balance_classes

    def next_class(self):
        self.current_class = (self.current_class + 1) % self.nclasses

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = np.random.randint(len(self.data))
        vid, label, dur, nf, pfl = self.data[index]
        if self.balance_classes:
            while len(np.where(pfl == self.current_class)[0]) == 0 or len(pfl) - self.window_size < 1:
                index = np.random.randint(len(self.data))
                vid, label, dur, nf, pfl = self.data[index]
            start_f = np.random.choice(np.where(pfl == self.current_class)[0])
        else:
            while len(pfl) - self.window_size < 1:
                index = np.random.randint(len(self.data))
                vid, label, dur, nf, pfl = self.data[index]
            start_f = np.random.randint(len(pfl)-self.window_size)
        end_f = min(start_f + self.window_size, nf)
        label = label[:, start_f:end_f]

        self.next_class()
        features = np.load(os.path.join(self.root, vid, f'i3d_features_{start_f}-{end_f}.npy'))

        return torch.from_numpy(features), torch.from_numpy(label)

    def __len__(self):
        return max(1000, len(self.data))


if __name__ == '__main__':
    # root = '/coc/pcba1/Datasets/public/mpii_cooking_2/videos_frames'
    root = '/coc/pcba1/dscarafoni3/NRI/NRI_Data/cropped_rgb_data/i3d_feature:q' \
           ''
    dataset = I3D_Dataset(os.path.join(root, '../config.json'), 'train', root, 5, False)
    ls = []
    for i in range(len(dataset.data)):
        d, l = dataset[i]
        ls.extend(l.data.numpy().argmax(axis=0).tolist())
    print(helpers.normalized_counter(ls))

