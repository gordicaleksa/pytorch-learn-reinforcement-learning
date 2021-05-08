import os


import numpy as np
import cv2 as cv
import imageio


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def create_gif(frames_dir, out_path, fps=30, img_width=None):
    assert os.path.splitext(out_path)[1].lower() == '.gif', f'Expected .gif got {os.path.splitext(out_path)[1]}.'

    frame_paths = [os.path.join(frames_dir, frame_name) for frame_name in os.listdir(frames_dir) if frame_name.endswith('.jpg')]

    images = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimwrite(out_path, images, fps=fps)
    print(f'Saved gif to {out_path}.')
