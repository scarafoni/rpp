import torch
import torch.utils.data as data_utl
import numpy as np
import json
import os
import os.path
import cv2
from Research_Platform import helpers
import traceback
import accimage


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


def image_to_np(image):
    """
    Returns:
        np.ndarray: Image converted to array with shape (width, height, channels)
    """
    image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
    image.copyto(image_np)
    image_np = np.transpose(image_np, (1, 2, 0))
    return image_np


def load_rgb_frame(data):
    image_dir, vid, i = data
    # try:
    # img = cv2.imread(os.path.join(image_dir, vid, 'frame' + str(i) + '.jpg'))[:, :, [2, 1, 0]]
    try:
        img = accimage.Image(os.path.join(image_dir, vid, f'frame{i:06d}.jpg'))
        # print(os.path.join(image_dir, vid, f'frame{i:06d}.jpg'))
    except Exception as e:
        track = traceback.format_exc()
        print(track)

    img = image_to_np(img)
    w, h, c = img.shape
    if w < 226 or h < 226:
        d = 226. - min(w, h)
        sc = 1 + d / min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
    img = (img / 255.) * 2 - 1
    return img


def load_rgb_frames(image_dir, vid, start, num):
    iis = list(range(start, start+num))
    image_dir = [image_dir]*len(iis)
    vid = [vid]*len(iis)
    data = list(zip(image_dir, vid, iis))
    frames = helpers.parmap(load_rgb_frame, data)
    try:
        f =np.asarray(frames, dtype=np.float32)
    except Exception as e:
        print([x.shape for x in frames])
        print(vid, start)

    return f


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


def make_dataset(split_file, split, root, mode, num_classes):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    vids = sorted(data.keys())
    for vid in vids:
        if data[vid]['split'] != split and not split == 'all':
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames // 2

        label = np.zeros((num_classes, num_frames), np.float32)

        # fps = num_frames / data[vid]['duration']
        for ann in zip(data[vid]['actions'], data[vid]['start_frames'], data[vid]['end_frames']):
            for fr in range(ann[1], min(ann[2]+1, num_frames)):
                label[ann[0], fr] = 1  # binary classification
        dataset.append((vid, label, num_frames*30., num_frames))
        i += 1

    return dataset


class I3D_Dataset(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, nclasses, transforms=None, window_size=64, chunksize=2000):

        self.data = make_dataset(split_file, split, root, mode, nclasses)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.window_size=window_size
        self.chunksize = chunksize

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        # index = np.random.randint(len(self.data))
        vid, label, dur, nf = self.data[index]
        if nf > self.chunksize:
            return os.listdir(os.path.join(self.root, vid)), torch.from_numpy(np.array([[-1]])), vid

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, 0, nf-1)
        else:
            imgs = load_flow_frames(self.root, vid, 0, nf-1)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # root = '/coc/pcba1/Datasets/public/mpii_cooking_2/videos_frames'
    root = '/coc/pcba1/Datasets/public/MSR-DailyActivities3D/frames'
    dataset = I3D_Dataset(os.path.join(root, '../config.json'), 'train', root, 'rgb', 32)
    print(len(dataset))
    for i in range(len(dataset.data)):
        d = dataset[i]
        print(d[-1])

