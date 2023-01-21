"""
file of useful stuff
"""
import warnings
import os
import random
from argparse import ArgumentParser
from os.path import join as opj
import shutil
import time
import numpy as np
import h5py
import json
import pickle
from collections import Counter
import re
import pathlib
import glob
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('pdf')
import traceback
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

_MODULE_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])

dataset_dir = '/coc/pcba1/Datasets/public'
ikeadb_dir = opj(dataset_dir, 'ikea-fa-release-data')


class Timer(object):
    """
    run this like to
    with Timer:
        <code>
    and it will time whatever is in the block
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))


class Dummy(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def delete_tree_if_exists(d):
    if os.path.exists(d):
        shutil.rmtree(d)


def delete_if_exists(f):
    if os.path.exists(f):
        os.remove(f)


def maybe_create_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(d):
    #     os.mkdir(d)


def maybe_create_fullfile_dir(*args):
    d = fullfile(*args)
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(d):
    #     os.mkdir(d)


def maybe_reset_director(d, reset):
    if reset:
        reset_directory(d)

def fixme(t):
    print(f"""********** FIXME **********
            {t}
********** FIXME **********""")

def reset_directory(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    mkdirp(d)


def clear_directory(d):
    files = glob.glob(f'{d}/*')
    for f in files:
        os.remove(f)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def fullfile(*args):
    return os.path.join(*args)


def suppress_warnings():
    warnings.filterwarnings("ignore")


def list_equal(l1, l2):
    eqs = []
    for ll1, ll2 in zip(l1, l2):
        eqs.append(ll1 == ll2)
    return all(eqs)


def get_all_files_by_extension(d, extension):
    """
    gets all teh files in a directory by the extension (recursive)
    :param d:  the driectory
    :return:
    """
    files_to_return = []
    for root, dirs, files in os.walk(d, followlinks=True):
        for name in files:
            if name.endswith(extension):
                files_to_return.append(opj(root, name))
    return files_to_return


def get_all_files_by_substring(d, substr):
    """
    gets all teh files in a directory that contain a substring (recursive)
    :param d:  the driectory
    :return:
    """
    files_to_return = []
    for root, dirs, files in os.walk(d):
        for name in files:
            if substr in name:
                files_to_return.append(opj(root, name))
    return files_to_return


def filename_without_extension(f):
    return os.path.splitext(os.path.basename(f))[0]


def basename_without_extension(f):
    return filename_without_extension(os.path.basename(f))


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    from stackoverflow
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


def most_common_in_row(arr):
    """
    gets the most common element in each row of a numpy array
    :param arr: array
    :return:
    """

    def f(a):
        counts = np.bincount(a)
        return np.argmax(counts)

    return np.apply_along_axis(f, 1, arr)


def scan_hdf5(path, recursive=True, tab_step=2):
    """
    opens hdf5 and print the interior
    :param path:
    :param recursive:
    :param tab_step:
    :return:
    """
    def scan_node(g, tabs=0):
        print(' ' * tabs, g.name)
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                print(' ' * tabs + ' ' * tab_step + ' -', v.name)
            elif isinstance(v, h5py.Group) and recursive:
                scan_node(v, tabs=tabs + tab_step)
    with h5py.File(path, 'r') as f:
        scan_node(f)


def most_common(lst):
    """
    gets the most common elemetn in list lst
    :param lst:
    :return:
    """
    if type(lst) == type(np.array([1])):
        lst = lst.tolist()
    return max(set(lst), key=lst.count)


def most_common_all(lst):
    """
    all the elements in a list sorted by most common with counts
    :param lst:
    :return:
    """

    if type(lst) == type(np.array([1])):
        lst = lst.tolist()

    c = Counter(lst)
    elements = []
    counts = []
    for key, value in c.items():
        elements.append(key)
        counts.append(value)

    counts = np.array(counts)
    elements = np.array(elements)

    ii = np.argsort(counts)[::-1]
    counts = counts[ii]
    elements = elements[ii]

    return elements, counts


def h5_keys(f):
    """ lists the keys of a given hyp5 file marker"""
    return list(f.keys())


def dump_json_pretty(data, file):
    """ dumps dictionary data to file in json format but prettified"""
    json.dump(data, open(file, 'w'), indent=4)


def load_json(file):
    return json.load(open(file, 'r'))


def pickleload(f, protocol=None):
    return pickle.load(open(f, 'rb'))


def picklesave(f, data, protocol=None):
    pickle.dump(data, open(f, 'wb'), protocol=protocol)


def unique(l):
    return list(set(l))


def normalized_counter(data, aslist=False):
    c = Counter(data)
    total = sum(c.values(), 0.0)
    for key in c:
        c[key] = np.round(c[key] /total, 3)

    if aslist:
        c = list([np.round(x, 3) for x in c.values()])
    return c


def find_substring_by_regex(s, ss):
    """
    finds and returns a substring from stack overflow:
    text = 'gfgfdAAA1234ZZZuijjk'

    m = re.search('AAA(.+?)ZZZ', text)
    """

    m = re.search(ss, s)
    return m.group(1)


def findall_regex(s, regex):
    p = re.compile(regex)
    return p.findall(s)


def pad_array_to_matching_length(a1, a2, mode='constant', padval=0):
    # pads a1 till it's as long as a2, assumes a batch and will pad second dimension
    return np.pad(a1, [0, len(a2)-len(a1)], mode=mode, constant_values=(padval, padval))


def pad_array_to_len(a, l, padval=0):
    return np.pad(a, [[0, l-len(a)]] + [[0, 0] for _ in range(len(a.shape[1:]))], mode='constant', constant_values=(padval, padval))


def add_batch(x):
    return np.expand_dims(x, 0)


def assert_ndims(x, n):
    # asserts that x has n dims
    assert(len(x.shape) == n)


def assert_all_equal_len(l):
    assert(all([len(l[0]) == len(ll) for ll in l]))


def extract_xdiff(X, abs=True):
    """
    extracts velcity of acceleration from X data (differentiations)
    :param X:
    :return: matrix of differences accross X points
    """
    Xv = np.zeros_like(X)
    for i in range(1, Xv.shape[0]):
        if abs:
            Xv[i] = np.abs(X[i] - X[i-1])
        else:
            Xv[i] = X[i] - X[i-1]

    return Xv


def median_pool(arr, k=5):
    vs = []
    for starti in range(len(arr)):
        endi = np.min([len(arr), starti + k])
        v = np.median(arr[starti:endi])
        vs.append(v)
    vs = np.array(vs)
    return vs


def readlines(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def read_file(fname):
    with open(fname, 'r') as f:
        lines = f.read()
    return lines

def write_file(fname, contents):
    with open(fname, 'w') as f:
        f.write(contents)

def listdir_full(folder):
    # lists all files and folders with the full path in a directory
    elements = []
    for dirname, dirnames, filenames in os.walk(folder):
         # print path to all filenames.
        for filename in filenames:
            elements.append(os.path.join(dirname, filename))       # print path to all filenames.

        for filename in filenames:
            elements.append(os.path.join(dirname, filename))

    return elements


def maxpool(arr, k=5):
    vs = []
    for starti in range(len(arr)):
        endi = np.min([len(arr), starti + k])
        v = np.max(arr[starti:endi])
        vs.append(v)
    vs = np.array(vs)
    return vs


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling
    https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy
    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'min':
        return A_w.min(axis=(1,2)).reshape(output_shape)


def convert_unsupervised_labels(y, y_):
    """
    in clustering, the assigned labels are supposed to match up with an original, but might not cover the
    exact same labels e.g. label 0 might be written as label 1 in y_. this converts y_ to y
    by running through y_ (in order of most common ys), converting to the most common label in y
    :param y: tru labels
    :param y_: guessed labels
    :return:
    """
    mapping = {}
    y__sorted, y__counts = most_common_all(y_)
    used_ys = []
    for label in y__sorted:
        corresponding_ys = y[y_ == label]
        l, _ = most_common_all(corresponding_ys)
        for ll in l:
            if not ll in used_ys:
                break
        mapping[label] = ll
        used_ys.append(ll)

    y__ = np.zeros_like(y_)
    for i, yy in enumerate(y_):
        y__[i] = mapping[yy]

    return y__, mapping


def dict_keys_items_in_order(d):
    # returns the keys, vals of d in a list in order
    keys = list(d.keys())
    vals = [d[k] for k in keys]

    return keys, vals


def smooth_array(x, N=10):
    # from stackoverflow
    return np.convolve(x, np.ones((N,)) / N, mode='same')


def smooth_array(x, N=10):
    # from stackoverflow
    return np.convolve(x, np.ones((N,)) / N, mode='same')


def index_percent_into(arr1, i, arr2):
    """
    given that we are at index i in arr1, return an index into arr2 that is the same percent of the way in
    """
    idx = int(i/len(arr1)*len(arr2))
    return idx


def arr_minmax(a):
    return np.min(a), np.max(a)


def where1d(cond):
    # like np.where but for one dimension(99 % of use cases)
    return np.where(cond)[0]


def normalize_rows(a):
    # normalize the rows of a matrix in numpy so they add to 1
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]

    return new_matrix


def imshow(p, colorbar=False, title=None, size=None):
    if size is not None:
        fig = plt.figure(figsize=size)
    else:
        fig = plt.figure()

    pos = plt.imshow(p)
    if title is not None:
        plt.title(title)

    if colorbar:
        fig.colorbar(pos)
    fig.show()


def imsave(p, f):
    plt.imshow(p)
    plt.savefig(f)


def npmap(f, d):
    x = list(map(f, d))
    return np.array(x)


def parmap(f, d):
    pool = Pool(4)
    res = pool.map(f, d)
    pool.close()
    pool.join()
    return res


def lk(d):
    return list(d.keys())


def feid(d):
    return d[list(d.keys())[0]]


def smooth_list_sequence(sequence, window=5):
    """
    given a sequence of actions (letters) smoothe over it using maxpooling
    :param sequence:
    :return:
    """

    newsequence = []
    for starti in range(len(sequence)):
        endi = min(len(sequence), starti+window)
        grab = sequence[starti:endi]
        newsequence.append(most_common(grab))
    return newsequence


def list_pooler(sequence, time='nframes'):
    # returns teh shortened pools and the time (nframes) for each action sequence or the start and end frame
    # WARNING- I'm fairly certain these will clip the end action off of the series
    pooled = []
    times = []
    if time == 'nframes':
        prev = 999
        for s in sequence:
            if not s == prev:
                pooled.append(s)
                times.append(0)
                prev = s

            times[-1] += 1
    else:
        startt = 0
        prev = 999
        for i, s in enumerate(sequence):
            if not s == prev:
                endt = max(i-1, 0)
                if prev != 999:
                    pooled.append(prev)
                    times.append([startt, endt])
                    startt = endt+1
            prev = s
        pooled.append(prev)
        times.append([startt, i])
        assert(times[-1][1] == len(sequence)-1)

        assert(len(pooled) == len(times))
        for action, time in zip(pooled, times):
            # print(time, action, sequence[time[0]:time[1]+1])
            assert(len(np.unique(sequence[time[0]:time[1]+1])) == 1)
            assert(np.unique(sequence[time[0]:time[1]+1])[0] == action)

    return pooled, times


def reverse_list_pooler(sequence, times):
    """
    converts a list of sequences and times for each into a per-frame list
    :param sequence: list of values
    :param times: number of frames for each value
    :return: list of values
    """

    final = []
    if len(sequence) != len(times):
        print(len(sequence),sequence)
        print(len(times), times)
        assert(len(sequence) == len(times))

    for s, t in zip(sequence, times):
        for tt in range(t):
            final.append(s)
    if len(final) != sum(times):
        print('final', final)
        print('times', times)
        print('sum v len', len(final), sum(times))
        assert(len(final) == sum(times))
    return final


def mkdirp(d):
    os.makedirs(d, exist_ok=True)


def channels_first_to_last(x):
    """
    changes the image x so that the channels are last, not first
    :param x:
    :return:
    """

    return np.rollaxis(x, 0, 3)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def accuracy(y_, y):
    if len(y_.shape) > 1:
        y_ = y_.argmax(axis=1)

    if len(y.shape) > 1:
        y = y.argmax(axis=1)
    return np.sum(y_ == y)/len(y)


def l2normalize(v):
    return v/np.linalg.norm(v)


def index_to_onehot(y, nclasses):
    b = np.zeros((len(y), nclasses))
    # print(y)
    b[np.arange(len(y)), y] = 1
    return b


def onehot_to_ind(y):
    return np.argmax(y, axis=1)


def reset_random_elements(seed=123):
    np.random.seed(seed)
    random.seed(seed)

def boiler_plate(gpu=0):
    gpu = int(gpu)
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        gpu = devices[gpu]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def balanced_data_generator(X, y):
    """
    generates data compensating for class imbalances
    :param X:
    :param y: assumed to be one hot
    :return:
    """
    c = -1
    yind = onehot_to_ind(y)
    while True:
        c = (c + 1) % 5
        idx = np.random.choice(np.where(yind == c)[0])
        yield X[idx], y[idx]


def weighted_acc(y, y_, w):
    return np.sum((y == y_)*w)/np.sum(w)


def next_experiment_dir(d, global_dir=False, nolog=False):
    """
    given an experimet dir with multiple trials of each experiment (each labeled with a number) select the next one
    :param d: directory with trial directories only within
    :return: path to newly created folder
    """

    if global_dir:
        s = '../../experiments'
    else:
        s = 'logs'
    maybe_create_dir(s)
    d = os.path.join(s, d)
    maybe_create_dir(d)

    trialdirs = [int(x) for x in os.listdir(d) if x.isdigit()]
    if len(trialdirs) == 0:
        last_trial = 0
        newdir = os.path.join(d, str(last_trial+1))
    else:
        for i in range(1, max(trialdirs)+1):
            if i not in trialdirs:
                newdir = os.path.join(d, str(i))
                break
        else:
            last_trial = max(trialdirs)
            newdir = os.path.join(d, str(last_trial+1))
    return newdir


def try_make_next_experiment_dir(d, slurm=False, nolog=False):
    if slurm:
        d = d + f'_{os.environ["SLURM_ARRAY_TASK_ID"]}'
        d = next_experiment_dir(d, nolog=nolog)
        maybe_create_dir(d)
        return d
    else:
        i = 0
        while True:
            try:
                d = next_experiment_dir(d, nolog=nolog)
                maybe_create_dir(d)
                break
            except Exception as e:
                i += 1
                print(f'could not get directory {d}')
                print(traceback.format_exc())
                if i > 1000:
                    print('error no way to get the file dirrectory')
                    break

            time.sleep(1)
        if nolog:
            d = '/'.join(d.split('/')[1:])
        return d


def latest_experiment_dir(d, slurm=False, old=False, nolog=False):
    """
    given an experimet dir with multiple trials of each experiment (each labeled with a number) select the most recent one
    :param d: directory with trial directories only within
    :return: path to newly created folder
    """

    if not nolog:
        d = os.path.join('logs', d)
    if slurm:
        d = d + f'_{os.environ["SLURM_ARRAY_TASK_ID"]}'
        if old:
            return d
    trialdirs = [int(x) for x in os.listdir(d) if x.isdigit()]
    last_trial = max(trialdirs)
    return os.path.join(d, str(last_trial))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          figsize=(24, 12),
                          fontsize=20):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        # print("Normalized confusion matrix")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",fontsize=int(fontsize*.5),
                    color="white" if cm[i, j] > thresh else "black")

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(int(fontsize*.75))
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(int(fontsize*.75))

    fig.tight_layout()
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.title.set_fontsize(fontsize)
    return fig, ax, cm

def rl(x):
    return range(len(x))

def window_label_to_per_frame(l):
    """
    given a series of interval labels (e.g. [1 64 4]) instead return the list as one label per frame
    asserts that all frames have an action in them
    :param l: list of interval labels, every row is [startframe, endframe, label]
    :return: list of per-frame action labels
    """

    labels = []
    for starti, endi, action in l:
        for i in range(starti, endi+1):
            labels.append(action)

    return labels


def per_frame_to_window_label(labels):
    """
    given a list of per-frame labels, convert them to intervals
    """

    label_list, times = list_pooler(labels, 'intervals')
    start_times = [x[0] for x in times]
    end_times = [x[1] for x in times]
    i = 0
    checker, frames = list_pooler(labels)
    assert(tuple(checker) == tuple(label_list))
    for k, (label, start_time, end_time) in enumerate(zip(label_list, start_times, end_times)):
        assert(frames[k] == (end_time-start_time)+1)
        for j in range(start_time, end_time+1):
            assert(label == labels[i])
            i += 1
    assert(i == len(labels))
    assert(labels[0] == label_list[0])
    assert(labels[-1] == label_list[-1])

    return label_list, start_times, end_times


def lzip(*args):
    return list(zip(*args))


def pad_window_labels(start_frames, end_frames, actions, n_frames, null_class):
    """
    given a list of labels over windows of the form [startframe, endframe, label] return a list with null action pads between
    gaps int he system such that the entire list spans all frames
    :param labels: zip of start_frames, end_frames, and actions
    :param n_frames:
    :param null_class:
    :return:
    """

    labels = lzip(start_frames, end_frames, actions)
    padded_labels = []
    starti = 0
    for listi, (start_frame, end_frame, action) in enumerate(labels):
        if starti < start_frame:
            padded_labels.append([starti, start_frame-1, null_class])
        padded_labels.append([start_frame, end_frame, action])
        starti = end_frame+1

    if labels[-1][1] < n_frames-1:
        padded_labels.append([labels[-1][1]+1, n_frames-1, null_class])

    for i, labels in enumerate(padded_labels[1:], start=1):
        # print(padded_labels[i][0], padded_labels[i-1][1]+1)
        assert(padded_labels[i][0] == padded_labels[i-1][1]+1)
    assert(padded_labels[-1][1] == n_frames-1)

    return lzip(*padded_labels)


def train_test_val_split(data, test_size=.3, val_size=.2):
    """
    splits data into train, test, and val sets
    :param test_size: percent of data you want to be test data
    :param val_size: percent of train split you want to leave for val
    :return: Xt, Xv, Xe, yt, yv, ye
    """

    Xt, Xe, yt, ye = train_test_split(*data, test_size=test_size)
    Xt, Xv, yt, yv = train_test_split(Xt, yt, test_size=val_size)

    return Xt, Xv, Xe, yt, yv, ye


def divide_videos_by_partition(partitions, outdir, verbose=False):
    """
    given the list of videos invideos, a directory to put them in, and a dictionary mapping partition to videos, divide the video into folders
    :param outdir: directory to dump farha_cc_results in
    :param partitions: dictoinary mapping partition names to video names (to be copied)
    :return:
    """

    for partition, videos in partitions.items():
        maybe_create_dir(opj(outdir, partition))
        for video in videos:
            basename = basename_without_extension(video)
            vname = os.path.basename(video)
            if os.path.exists(opj(outdir, partition, f'{basename}.avi')):
                if verbose: print(f'skipping- {video}- file exists')
                continue

            t = opj(outdir, partition, vname)
            if verbose: print(f'copying {video} to {t}')
            shutil.copy(video, t)


def nth_last_folder(path, n):
    # returns the nth last folder in path e.g. 2 is the folder up from the final folder
    split = path.split('/')[:-1*n]
    return '/'.join(split)


def sliding_window_intervals(tot_len, window_size, stride=1, full_size_only=False):
    """
    returns start, end idexes into chunks of frm 0 to l-1 of size c
    :param tot_len: the total length of the sequence
    :return:
    """

    ls = []
    for i in range(0, tot_len-1, stride):
        endi = min(tot_len-1, i + window_size)
        ls.append([i, endi])

    if full_size_only:
        return [l for l in ls if l[1] - l[0] == window_size]
    else:
        return ls


def exists(f):
    return os.path.exists(f)


def printable_progress_bar(x, n):
    output = ['_']*100
    for i in range(int(x/n*100)-1):
        output[i] = '#'

    return ''.join(output)


bp_parser = ArgumentParser()
bp_parser.add_argument('logdir', type=str)
bp_parser.add_argument('--gpu', type=int, nargs='+', default=[0])
bp_parser.add_argument('--verbose', action='store_true')
bp_parser.add_argument('--debug', action='store_true')
bp_parser.add_argument('--slurm', action='store_true')
bp_parser.add_argument('--eval-only', type=int, default=0)
bp_parser.add_argument('--seed', type=int, default=123)
bp_parser.add_argument('--batch-size', type=int, default=128)

bp_parser_nologdir = ArgumentParser()
bp_parser_nologdir.add_argument('--logdir', type=str, default='test')
bp_parser_nologdir.add_argument('--gpu', type=int, nargs='+', default=[0])
bp_parser_nologdir.add_argument('--verbose', action='store_true')
bp_parser_nologdir.add_argument('--debug', action='store_true')
bp_parser_nologdir.add_argument('--slurm', action='store_true')
bp_parser_nologdir.add_argument('--eval-only', type=int, default=0)
bp_parser_nologdir.add_argument('--seed', type=int, default=123)
bp_parser_nologdir.add_argument('--batch-size', type=int, default=128)

processing_parser = ArgumentParser()
processing_parser.add_argument('--reset', action='store_true')
processing_parser.add_argument('--batch-size', type=int, default=128)
processing_parser.add_argument('--gpu', type=int, default=0)
processing_parser.add_argument('--verbose', action='store_true')

def quickplot(x, name='test.png'):
    f = plt.figure()
    for xx in x:
        plt.plot(xx)
    plt.savefig(name)
<<<<<<< HEAD
    plt.close(f)

def fixme(s):
    print('### FIXME ###')
    print(s)
    print('### FIXME ###')
=======
    plt.close(f)
>>>>>>> 9f2d2be6156b61abae867f021925bef52546f85d
