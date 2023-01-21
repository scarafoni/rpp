"""
given video input and some start/end frames and actions, write the actions onto the video
"""

from Research_Platform import helpers
from skvideo.io import vwrite
import cv2 as cv
import numpy as np
import os

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
    config = helpers.load_json(os.path.join(config))
    vid = config[vname]
    frames, actions = unfold_frames(vid, activities=activities)
    cap = cv.VideoCapture(video_file)
    i = 0
    processed_frames = []
    while True:
        ret, frame = cap.read()
        # cv.imshow(actions[i], frame)

        if cv.waitKey(1) & 0xFF == ord('q') or ret is False:
            break

        frame = cv.putText(frame, f'current action- {actions[i]}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        processed_frames.append(frame[..., [2,1,0]])

        i += 1
        if i == len(actions)-1:
            break
        # if i > 1000:
        #     break

    cap.release()
    cv.destroyAllWindows()
    processed_frames = np.array(processed_frames)
    vwrite(f'{vname}_output.avi', processed_frames)


if __name__ == "__main__":
    write_action_to_video()
