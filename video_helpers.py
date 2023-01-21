import cv2 as cv
import numpy as np
import time
try:
    from . import helpers
except Exception as e:
    from Research_Platform import helpers
import os
from skvideo.io import vread, vwrite
from skimage.io import imread
import re
import os

def maybe_read_video(video):
    """
    opens a video from a string if the input is a string
    :param video:
    :return:
    """

    if type(video) == str:
        video = vread(video)
    return video


def flow_video(flows):
    """
    takes in optical flow (flow) and outputs it as an mp4 in fout
    :param flow: list of flows each of whic is hxwx2 optical flow
    :param fout: mp4 file to save to
    :return: npy of optical flow
    """

    rgbs = []

    for flow in flows:
        print(list(flow.shape[:-1]) + [3])
        hsv = np.zeros(list(flow.shape[:-1]) + [3])
        hsv[..., 1] = 225
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        print(hsv.shape)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        rgbs.append(np.expand_dims(rgb, 0))

    end_rgb = np.vstack(rgbs)
    return end_rgb


def watch_video(fname):
    cap = cv.VideoCapture(fname)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def resize_video(fin, fout, out_size=(340, 256), inplace=False):
    """
    resizes a video
    :param fin: input videoname or video- video only works for in place
    :param fout: output video name
    :param out_size: outsize
    :param inplace- wehterh to return the resized video
    :return:
    """

    if inplace:
        if type(fin) == str or type(fin) == np.str_:
            video = vread(fin)
        else:
            video = fin
        video2 = np.zeros([len(video), *out_size, 3])
        for i in range(len(video)):
            video2[i] = cv.resize(video[i], (out_size[1], out_size[0]))
        return video2
    else:
        os.system(f'ffmpeg -y -loglevel panic -i {fin} -s {out_size[0]}x{out_size[1]} -c:a copy {fout}')


def resize_video_folder(input_folder, output_folder, out_size=(340, 256)):
    """
    resiezes all teh videos in a folder
    :param input_folder:
    :param output_folder:
    :return: n/a
    """

    helpers.reset_directory(output_folder)

    for f in helpers.listdir_fullpath(input_folder):
        fname = os.path.basename(f)
        try:
            resize_video(f, os.path.join(output_folder, fname), out_size=out_size)
        except Exception as e:
            print(f'error getting video on {f}- {e}')


def dump_video_cv2(filename, clip, fourcc_str='MJPG', fps=30.0):
    """Write video on disk from a stack of images
    from https://www.programcreek.com/python/example/72134/cv2.VideoWriter
    Parameters
    ----------
    filename : str
        Fullpath of video-file to generate
    clip : ndarray
        ndarray where first dimension is used to refer to i-th frame
    fourcc_str : str
        str to retrieve fourcc from opencv
    fps : float
        frame rate of create video-stream

    """
    fourcc = cv.VideoWriter_fourcc(*fourcc_str)
    fid = cv.VideoWriter(filename, fourcc, fps, clip.shape[1:3][::-1])
    if fid.isOpened():
        for i in range(clip.shape[0]):
                fid.write(clip[i, ...])
        fid.release()
        return True
    else:
        return False


def crop_video(video, new_h=224, new_w=224, channels='tensorflow'):
    if len(video.shape) < 4:
        raise Exception(f'the video should have 4 dimensions, instead has {len(video.shape)}')

    # crops a video (4d numyp array)
    if channels == 'tensorflow':
        h, w = video.shape[1], video.shape[2]
    else:
        h, w = video.shape[2], video.shape[3]

    minh = int((h - new_h) / 2)
    maxh = minh + new_h
    minw = int((w - new_w) / 2)
    maxw = minw + new_w

    if channels == 'tensorflow':
        video = video[:, minh:maxh, minw:maxw, :]
    else:
        video = video[:, :, minh:maxh, minw:maxw]
    return video


def crop_video_to_box(vfile, outfile, boxes):
    """
    crops the video to a box determined by boxes
    :param vfile: the file of the video to read and crop
    :param boxes: list of boxes to do crops on, must be as long as the vfile. All boxes must be the same size
    :param outfile: the file to sve the new video to
    :return:
    """

    video = vread(vfile)
    newh = int(boxes[0, 3]) - int(boxes[0, 1])
    neww = int(boxes[0, 2]) - int(boxes[0, 0])

    cropped_video = np.zeros([len(boxes), int(newh), int(neww), 3])

    assert(len(video) == len(boxes))

    for i, (frame, box) in enumerate(zip(video, boxes)):
        sh = int(box[1])
        eh = int(box[3])
        sw = int(box[0])
        ew = int(box[2])
        cropped_video[i] = frame[sh:eh, sw:ew]

    cropped_video = cropped_video.astype(np.uint8)
    vwrite(outfile, cropped_video)
    # dump_video_cv2(outfile, cropped_video)


def to_channels_first(v):
    # takes a FxHxWxC matrix and returns it as FxCxHxW
    return np.rollaxis(v, 3, 0)


def resize_frame_with_aspect_ratio(frame, shorter_side=256):
  """
  Resize a frame using OpenCV.
  :param frame:           A single video frame.
  :param shorter_side:    Size of the target shorter side, longer side will be computed so that the aspect ratio
                          is preserved.
  :return:                Resized frame.
  """

  if frame.shape[0] > frame.shape[1]:
    long = frame.shape[0]
    short = frame.shape[1]
  else:
    short = frame.shape[0]
    long = frame.shape[1]

  fract = shorter_side / short
  target_long = int(long * fract)

  if frame.shape[0] > frame.shape[1]:
    return cv.resize(frame, (shorter_side, target_long))
  else:
    return cv.resize(frame, (target_long, shorter_side))

def frame_folder_to_video(frame_folder, out_video, framesize=(640, 480)):
    """
    converts a folder of frames to a video (avi) file
    :param frame_folder:
    :param out_video:
    :return:
    """

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    writer = cv.VideoWriter(out_video, fourcc=fourcc, fps=30, frameSize=framesize)

    if not writer.isOpened():
        return False

    frames = get_frame_files_from_folder(frame_folder)
    for frame in frames:
        frame = cv.resize(frame, framesize)
        writer.write(frame)

    writer.release()


def video_to_jpgs(video_path, save_path, do_resize=True, shorter_side=256):
  """
  taken from the kinetics_downloader github repo
  Extract individual frames from a video.
  :param video_path:          Path to the video file.
  :param save_path:           Path to a directory where to save the video frames.
  :param do_resize:           Resize the frames.
  :param shorter_side:        If do_resize, shorter side will be resized to this value.
  :return:                    True if extraction successful, otherwise false.
  """

  cap = cv.VideoCapture(video_path)

  if not cap.isOpened():
    return False

  i = 0
  res, frame = cap.read()

  spath = os.path.join(save_path, helpers.basename_without_extension(video_path))
  helpers.reset_directory(spath)

  while res:
    if do_resize:
      frame = resize_frame_with_aspect_ratio(frame, shorter_side=shorter_side)
    cv.imwrite(os.path.join(spath, "frame{:06d}.jpg".format(i)), frame, [int(cv.IMWRITE_JPEG_QUALITY), 75])
    res, frame = cap.read()
    i += 1

  num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  num_images = len(os.listdir(spath))

  return num_frames == num_images


def flow_to_jpgs(flow_path, save_path, do_resize=True, shorter_side=256):
    """
    taken from the kinetics_downloader github repo
    Extract individual frames from a video.
    :param flow_path:          Path to the flow numpy.
    :param save_path:           Path to a directory where to save the video frames.
    :param do_resize:           Resize the frames.
    :param shorter_side:        If do_resize, shorter side will be resized to this value.
    :return:                    True if extraction successful, otherwise false.
    """

    frames = np.load(flow_path)

    spath = os.path.join(save_path, helpers.basename_without_extension(flow_path))
    helpers.reset_directory(spath)

    for i, frame in enumerate(frames):
        if do_resize:
          frame = resize_frame_with_aspect_ratio(frame, shorter_side=shorter_side)
        np.save(os.path.join(spath, "frame{:d}.npy".format(i)), frame)

    num_frames = len(frames)
    num_images = len(os.listdir(spath))

    return num_frames == num_images


def folder_to_jpgs(input_folder, output_folder, flow=False, extension='.avi', verbose=False, reset=False):
    """
    converts all teh videos in a folder to jpgs
    :param input_folder:
    :param output_folder:
    :return: n/a
    """

    if reset:
        helpers.reset_directory(output_folder)

    for f in helpers.get_all_files_by_extension(input_folder, extension):
        if verbose:
            print(f)
        hold = []
        middle_folder = ''
        for i, x in enumerate(f.split('/')):
            hold.append(x)
            if '/'.join(hold) == input_folder:
                middle_folder = '/'.join(f.split('/')[i+1:-1])
                break
        try:
            if flow:
                flow_to_jpgs(f, os.path.join(output_folder, middle_folder), False)
            else:
                video_to_jpgs(f, os.path.join(output_folder, middle_folder), False)
        except Exception as e:
            print(f'error getting video on {f}- {e}')


def frame_num_from_jpgs(f):
    return int(re.findall(r'\d+', f)[-1])


def get_frame_files_from_folder(folder):
    frames = helpers.get_all_files_by_extension(folder, '.jpg')
    frames = sorted(frames, key=lambda x: frame_num_from_jpgs(x))
    return frames


def get_video_length_from_frame_folder(folder):
    frames = get_frame_files_from_folder(folder)
    return frame_num_from_jpgs(frames[-1])


def frame_from_folder_by_num(folder, n):
    return imread(os.path.join(folder, f'frame{n}.jpg'))


def get_video_folders(folder):
    # returns the video folders in a directory
    frames = get_frame_files_from_folder(folder)
    folder_names = ['/'.join(x.split('/')[:-1]) for x in frames]
    folders = helpers.unique(folder_names)
    return folders


def get_video_array_from_folder(folder):
    fnames = get_frame_files_from_folder(folder)
    imgs = [imread(x) for x in fnames]
    return np.array(imgs)


def get_video_metadata(video):
    """ gets the fps, lenth, nframes, and size of a video without opening it"""

    assert(os.path.exists(video))
    cap = cv.VideoCapture(video)
    fps = cap.get(cv.CAP_PROP_FPS)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = nframes / fps
    cap.release()

    return (nframes, fps, duration, size)


def frames_from_vfile(video):
    return get_video_metadata(video)[0]


def vlen_from_vfile(video):
    return get_video_metadata(video)[2]


# def video_len(v):
#     # do not use
#     return get_video_metadata(v)[0]


def convert_video_fps(infile, outfile, fps):
    # converts video infile to video outfile with fps fps
    os.system(f'ffmpeg -i {infile} -filter:v fps=fps={fps} {outfile}')


def get_chunk_of_video(infile, outfile, startt, endt):
    # gets a video chunk from startt to endt
    os.system(f'ffmpeg -i {infile} -ss {startt} -to {endt} -c:v copy -c:a copy {outfile}')


def label_video_frames(vin_file, vout_file, labels):
    """
    labels a video with actions
    :param vin_file: video file
    :param vout_file: output file to save
    :param labels: list of actions
    :param frames: list of frames coresponding to each action
    :return: the video with action information written on it
    """

    cap = cv.VideoCapture(vin_file)
    nframes, fps, _, framesize = get_video_metadata(vin_file)
    while len(labels) < nframes:
        labels.append('n/a')
    assert(nframes == len(labels))
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    writer = cv.VideoWriter(vout_file, fourcc=fourcc, fps=fps, frameSize=framesize)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        else:
            font = cv.FONT_HERSHEY_SIMPLEX
            bottom = (10, int(framesize[1]*.75))
            fontScale = 1
            fontColor = (0, 255, 255)
            lineType = 2

            cv.putText(frame, str(labels[i]),
                       bottom,
                       font,
                       fontScale,
                       fontColor,
                       lineType)

        frame = cv.resize(frame, framesize)
        writer.write(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        i += 1

    cap.release()
    writer.release()

    return None


def videonames_to_clips(video_names, output_folder, video_actions, nclasses, reset=False, verbose=False):
    """
    given a folder of videos, break each entry into a series of clips based on action
    :param video_names: list of video names
    :param output_folder: folder to place clips in
    : video_actions: list of lists of actions, in the same order as teh video names, note there must be one action per frame
    for the videos
    :return: nothing
    """

    if not os.path.exists(output_folder):
        helpers.mkdirp(output_folder)
    if reset:
        helpers.reset_directory(output_folder)
    for i in range(0, nclasses):
        helpers.maybe_create_dir(os.path.join(output_folder, str(i)))

    for video, action in zip(video_names, video_actions):
        vbase = helpers.basename_without_extension(video)
        if verbose: print(f'on video- {vbase}')
        video_data = vread(video)
        if verbose: print(len(video_data), len(action))
        pooled_actions, frames = action_pooler(action, 'framecount')
        for a, f in zip(pooled_actions, frames):
            vd = video_data.copy()[f[0]:f[1]]
            if f[1] - f[0] < 1:
                continue
            if verbose: print(f'saving action {a} from frames {f[0]}-{f[1]} total len = {len(vd)}')
            v2write = f'{output_folder}/{a}/{vbase}_{f[0]}-{f[1]}.avi'
            if not os.path.exists(v2write):
                vwrite(v2write, vd)
            else:
                if verbose: print(f'skipping {v2write}, file exists')


def action_pooler(sequence, time='nframes'):
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

        assert(len(pooled) == len(times))
        for action, time in zip(pooled, times):
            # print(time, action, sequence[time[0]:time[1]+1])
            assert(len(np.unique(sequence[time[0]:time[1]+1])) == 1)
            assert(np.unique(sequence[time[0]:time[1]+1])[0] == action)

    return pooled, times


def apply_kinect_skeleton(frame, pose):
    lines = np.array([
        [0,1],
        [1,2],
        [2,3],
        [2,4],
        [4,5],
        [5,6],
        [6,7],
        [2,8],
        [8,9],
        [9,10],
        [10,11],
        [0,12],
        [0,16],
        [12,13],
        [13,14],
        [14,15],
        [0,16],
        [16, 17],
        [17,18],
        [18,19]
    ])

    for i, joint in enumerate(pose):
        # frame = cv.putText(frame, str(i), int(joint[0]), int(joint[1]+5), cv.FONT_HERSHEY_COMPLEX, (0,  255, 0))
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = .5
        fontColor = (0, 255, 255)
        lineType = 2

        frame = cv.circle(frame, center=(int(joint[0]), int(joint[1])), radius=10, color=(0, 255, 0), thickness=-1)
        # cv.putText(frame, str(i),
        #            (int(joint[0]+50), int(joint[1]+5)),
        #            font,
        #            fontScale,
        #            fontColor,
        #            lineType)

    for line in lines:
        start = tuple(pose[line[0]])
        start = tuple([int(x) for x in start])
        end = tuple(pose[line[1]])
        end = tuple([int(x) for x in end])
        frame = cv.line(frame, start, end, (0, 255, 0), 10)

    return frame


def bgr2rgb(frame):
    return frame[..., [2,1,0]]


def write_config_actions_to_video(config, vname, infile, outfile):
    """
    given a config and a video key name, write all the actions for that video on the frames and save the output video
    :param config: config json to use
    :param vname: vname of the video in the json
    :param infile:  video file to read in
    :param outfile:  video file to write it to
    :return: n/a
    """

    def unfold_frames(vid):
        frames = []
        actions = []
        # for i in range(vid['start_frames'][0]):
        #     actions.append('none')
        for i in range(len(vid['start_frames'])):
            action = vid['actions'][i]
            start_frame = vid['start_frames'][i]
            end_frame = vid['end_frames'][i]
            for j in range(start_frame, end_frame):
                actions.append(str(action))
                frames.append(j)
            # if i < len(vid['start_frames']) - 1:
            #     for j in range(end_frame, vid['start_frames'][i+1]):
            #         actions.append('none')
        return frames, actions

    # config = helpers.load_json(helpers.fullfile(data_dir, 'config_1.json'))
    config = helpers.load_json(config)
    vid = config[vname]
    frames, actions = unfold_frames(vid)
    cap = cv.VideoCapture(infile)
    i = 0
    processed_frames = []
    while True:
        ret, frame = cap.read()
        # cv.imshow(actions[i], frame)

        if cv.waitKey(1) & 0xFF == ord('q') or ret is False:
            break

        frame = cv.putText(frame, f'current action- {actions[i]}', (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        processed_frames.append(frame[..., [2,1,0]])

        i += 1
        if i == len(actions)-1:
            break
        # if i > 1000:
        #     break

    cap.release()
    cv.destroyAllWindows()
    processed_frames = np.array(processed_frames)
    vwrite(outfile, processed_frames)


def unfold_frames(vid, activities):
    frames = []
    actions = []
    for i in range(len(vid['start_frames'])):
        action = vid['actions'][i]
        start_frame = vid['start_frames'][i]
        end_frame = vid['end_frames'][i]
        for j in range(start_frame, end_frame):
            actions.append(activities[action])
            frames.append(j)
    return frames, actions


def write_action_to_video(video_file, vname, config, activities):
    vid = config[vname]
    frames, actions = unfold_frames(vid, activities=activities)
    cap = cv.VideoCapture(video_file)
    i = 0
    processed_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # if cv.waitKey(1) & 0xFF == ord('q') or ret is False:
        #     break

        frame = cv.putText(frame, f'current action- {actions[i]}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        processed_frames.append(frame[..., [2,1,0]])

        i += 1
        if i == len(actions)-1:
            break
        # if i > 1000:
        #     break

    cap.release()
    # cv.destroyAllWindows()
    processed_frames = np.array(processed_frames)
    vwrite(f'{vname}_output.avi', processed_frames)


if __name__ == '__main__':
    d = '../../data/external/8-17-data'
    for i in range(1, 8):
        write_config_actions_to_video(helpers.fullfile(d, 'overhead_config.json'), f'trial{i}', helpers.fullfile(d, 'overhead_videos', f'trial{i}.avi'), f'trial{i}.avi')

