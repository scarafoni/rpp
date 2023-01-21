"""
reads in resized videos, breaks them into arrays, and then extracts optical flow on them
"""
from .repos.tfoptflow.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from copy import deepcopy
from .repos.tfoptflow.optflow import flow_to_img
from skvideo.io import vwrite, vread
import numpy as np
from . import helpers, video_helpers
import os


def generate_net(output_size=(224,224), batch_size=128, gpu=0):
    gpu_devices = [f'/device:GPU:{0}']
    controller = f'/device:GPU:{0}'

    ckpt_path = '/coc/pcba1/dscarafoni3/Research_Platform/repos/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

    # Configure the model for inference, starting with the default options
    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = ckpt_path
    nn_opts['batch_size'] = batch_size
    nn_opts['gpu_devices'] = gpu_devices
    nn_opts['controller'] = controller

    # We're running the PWC-Net-large model in quarter-resolution mode
    # That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2

    # The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
    # of 64. Hence, we need to crop the predicted flows to their original size
    # nn_opts['adapt_info'] = (1, 436, 1024, 2)
    nn_opts['adapt_info'] = [1] + list(output_size) + [2]

    # Instantiate the model in inference mode and display the model configuration
    nn = ModelPWCNet(mode='test', options=nn_opts)
    return nn


def predict_from_video_list(frames, nn):
    """
    given a frame list, predicts flows
    :param frames:
    :return:
    """

    # resize the frames into pairs
    frame_pairs = []
    for i in range(len(frames)-1):
        frame_pairs.append((deepcopy(frames[i]), deepcopy(frames[i+1])))

    # Generate the predictions and display them
    pred_labels = nn.predict_from_img_pairs(frame_pairs, verbose=True)
    pred_labels = np.vstack([np.expand_dims(x, 0) for x in pred_labels])
    flow_vid = np.vstack([np.expand_dims(flow_to_img(x), 0) for x in pred_labels])
    return flow_vid, pred_labels


def quick_video_prediction(video, batch_size=32, gpu=0):
    if video.dtype == np.uint8:
        video = video / 255.
    nn = generate_net(video.shape[1:3], batch_size=batch_size, gpu=gpu)
    flow_video, flow_arr = predict_from_video_list(video, nn)
    return flow_video, flow_arr


def flow_for_folder(input_folder, output_folder, batch_size=64, gpu=0, resize=False, verbose=False):
    helpers.reset_directory(output_folder)
    files = helpers.listdir_fullpath(input_folder)
    if not resize:
        data = video_helpers.get_video_metadata(files[0])
        size = data[3]
    else:
        size = resize

    nn = generate_net(size, batch_size=batch_size, gpu=gpu)
    for f in files:
        # if os.path.exists(os.path.join(output_folder, f + '.npy')):
        #     print(f'skipping {fname}, exists')
        #     continue
        l = video_helpers.get_video_metadata(f)[0]
        if l < 3:
            continue
        if verbose:
            print(f'on file- {f}')
        if resize is not False:
            video = video_helpers.resize_video(f, 'none', size, inplace=True)
        else:
            video = vread(f)
        if video.dtype == np.uint8 or np.max(video) > 200:
            video = video / 255.
            video = video.astype(np.float32)

        flow_video, flow_arr = predict_from_video_list(video, nn)
        fname = helpers.filename_without_extension(f)
        np.save(os.path.join(output_folder, fname + '.npy'), flow_arr)
        vwrite(os.path.join(output_folder, fname + '.avi'), flow_video)


