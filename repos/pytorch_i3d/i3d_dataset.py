import torch
import torch.utils.data as data_utl
import sys
import numpy as np
import json
import os
import os.path
from Research_Platform import helpers, pytorch_helpers as ph
import cv2


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        # print(os.path.join(image_dir, vid, 'frame' + str(i) + '.jpg'))
        try:
            img = cv2.imread(os.path.join(image_dir, vid, 'frame' + str(i) + '.jpg'))[:, :, [2, 1, 0]]
        except Exception as e:
            print(e)
            print(f"error trying to get image {os.path.join(image_dir, vid, 'frame' + str(i) + '.jpg')}")
            sys.exit(1)
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes, window_size):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['split'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames // 2

        if num_frames < 66:
            continue

        label = np.zeros((num_classes, num_frames), np.float32)
        fps = num_frames / data[vid]['duration']
        for ann in zip(data[vid]['actions'], data[vid]['start_frames'], data[vid]['end_frames']):
            for fr in range(ann[1], min(ann[2]+1, num_frames-1)):
                # print(vid, ann, fr)
                label[ann[0], fr] = 1  # binary classification

        ys = np.argmax(label, axis=0)
        per_window_labels = []
        for j in range(0, len(ys)-window_size):
            endj = min(len(ys), j+window_size)
            l = ys[j:endj]
            per_window_labels.append(helpers.most_common(l))

        assert(len(per_window_labels) == len(ys)-window_size)
        per_window_labels = np.array(per_window_labels)

        dataset.append((vid, label, data[vid]['duration'], num_frames, per_window_labels))
        i += 1

    return dataset


class I3D_Dataset(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, nclasses, transforms=None, window_size=64, balance_classes=False):

        self.data = make_dataset(split_file, split, root, mode, nclasses, window_size)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.window_size=window_size
        self.balance_classes = balance_classes
        self.current_class = 0
        self.nclasses = nclasses

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        # index = np.random.randint(len(self.data))
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

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, self.window_size)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, self.window_size)

        label = label[:, start_f:start_f+self.window_size]
        assert(label.shape[1] == imgs.shape[0])
        if self.transforms is not None:
            imgs = self.transforms(imgs)

        imgs = video_to_tensor(imgs)
        return imgs, torch.from_numpy(label)

    def __len__(self):
        return len(self.data)*100


if __name__ == '__main__':
    # root = '/coc/pcba1/Datasets/public/mpii_cooking_2/videos_frames'
    root = '../../data/da3d/frames'
    dataset = I3D_Dataset(os.path.join('../../data/da3d/config.json'), 'train', root, 'rgb', 16)
    dl = ph.standard_dataloader(dataset)
    print(len(dataset))
    ls = []
    for d, l in dl:
        print(d.shape)
        ls.extend(l.data.numpy().argmax(axis=0).tolist())
    # print(helpers.normalized_counter(ls))

