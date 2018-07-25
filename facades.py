from paddle.dataset.image import *
import os
import glob
import paddle.v2 as paddle
import utils
import numpy as np
import random


# https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/dataset
def reader_creator(data_file, resize_size, crop_size, is_train, lab_colorization, which_direction, cycle=False):
    data_paths = glob.glob(os.path.join(data_file, "*.jpg"))

    if len(data_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in data_paths):
        data_paths = sorted(data_paths, key=lambda path: int(get_name(path)))
    else:
        data_paths = sorted(data_paths)

    def reader():
        while True:
            for data_path in data_paths:
                raw_image = load_image(file=data_path)  # raw_image has type: <type 'numpy.ndarray'>
                # split raw_image to input and target according to a.which_direction
                assert raw_image.shape[2] == 3

                if lab_colorization:
                    # load color and brightness from image, no B image exists here
                    lab = utils.rgb_to_lab(raw_image)
                    L_chan, a_chan, b_chan = utils.preprocess_lab(lab)
                    a_image = L_chan  # TODO:
                    b_image = np.stack([a_chan, b_chan], axis=2)
                else:
                    # break apart image pair and move to range [-1, 1]
                    width = raw_image.shape[1]  # [height, width, channels]
                    a_image = utils.preprocess(raw_image[:, :width // 2, :])
                    b_image = utils.preprocess(raw_image[:, width // 2:, :])

                if which_direction == "AtoB":
                    input, target = [a_image, b_image]
                elif which_direction == "BtoA":
                    input, target = [b_image, a_image]
                else:
                    raise Exception("invalid direction")

                input_image = simple_transform(im=input, resize_size=resize_size, crop_size=crop_size,
                                               is_train=is_train)
                target_image = simple_transform(im=target, resize_size=resize_size, crop_size=crop_size,
                                                is_train=is_train)

                # flip the images randomly
                r = random.random()
                if r > 0.5:
                    input_image = left_right_flip(input_image)
                    target_image = left_right_flip(target_image)

                yield input_image, target_image

            if not cycle:
                break

    return reader


def train(resize_size, crop_size, lab_colorization, which_direction):
    return reader_creator(data_file='./facades/train', resize_size=resize_size, crop_size=crop_size, is_train=True,
                          lab_colorization=lab_colorization, which_direction=which_direction)


def test(resize_size, crop_size, lab_colorization, which_direction):
    return reader_creator(data_file='./facades/test', resize_size=resize_size, crop_size=crop_size, is_train=False,
                          lab_colorization=lab_colorization, which_direction=which_direction)


def val(resize_size, crop_size, lab_colorization, which_direction):
    return reader_creator(data_file='./facades/val', resize_size=resize_size, crop_size=crop_size, is_train=False,
                          lab_colorization=lab_colorization, which_direction=which_direction)


def train_reader(resize_size, crop_size, batch_size, lab_colorization, which_direction):
    return paddle.batch(paddle.reader.shuffle(
        train(resize_size=resize_size, crop_size=crop_size, lab_colorization=lab_colorization,
              which_direction=which_direction), buf_size=500),
                        batch_size=batch_size)


def test_reader(resize_size, crop_size, batch_size, lab_colorization, which_direction):
    return paddle.batch(test(resize_size=resize_size, crop_size=crop_size, lab_colorization=lab_colorization,
                             which_direction=which_direction), batch_size=batch_size)


def val_reader(resize_size, crop_size, batch_size, lab_colorization, which_direction):
    return paddle.batch(val(resize_size=resize_size, crop_size=crop_size, lab_colorization=lab_colorization,
                            which_direction=which_direction), batch_size=batch_size)
