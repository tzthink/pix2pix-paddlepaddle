from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import argparse
import collections
import os
import numpy as np
import random
import utils
import math
import json
import time
from PIL import Image
from paddle.dataset.image import *
import facades
import glob
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--crop_size", type=int, default=256, help="cropping to 256x256 by default")

parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

parser.add_argument("--use_cuda", action="store_true", help="use cuda during the training")
parser.add_argument("--params_dirname", type=str, default="pix2pix.inference.model",
                    help="save parameter into a directory")
# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples_Meta = collections.namedtuple("Examples_Meta", "count, steps_per_epoch")
Model = collections.namedtuple("Model",
                               "discrim_trainer, gen_trainer")


def discrim_conv(batch_input, out_channels, stride=1, padding=0, filter_size=4):
    padded_input = fluid.layers.pad(batch_input, [0, 0, 0, 0, 1, 1, 1, 1])  # TODO:
    return fluid.layers.conv2d(input=padded_input, num_filters=out_channels, filter_size=filter_size, stride=stride,
                               padding=padding, act=None)


def gen_conv(batch_input, out_channels, stride=1, padding=0, filter_size=4):
    # [batch, in_channels, in_height, in_width] => [batch, out_channels, out_height, out_width]
    # TODO: paddle.networks.img_separable_conv
    return fluid.layers.conv2d(input=batch_input, num_filters=out_channels, filter_size=filter_size, stride=stride,
                               padding=padding, act=None)


def gen_deconv(batch_input, out_channels, stride=1, padding=0, filter_size=4):
    # [batch, in_channels, in_height, in_width] => [batch, out_channels, out_height, out_width]
    # TODO: paddle.networks.img_separable_conv
    return fluid.layers.conv2d_transpose(input=batch_input, num_filters=out_channels, filter_size=filter_size,
                                         stride=stride, padding=padding, act=None)


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    output = gen_conv(generator_inputs, a.ngf)
    layers.append(output)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        rectified = utils.lrelu(layers[-1], 0.2)
        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
        convolved = gen_conv(rectified, out_channels)
        output = utils.batchnorm(convolved)
        layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        if decoder_layer == 0:
            # first decoder layer doesn't have skip connections
            # since it is directly connected to the skip_layer
            input = layers[-1]
        else:
            input = fluid.layers.concat(input=[layers[-1], layers[skip_layer]], axis=1)

        rectified = fluid.layers.relu(input)
        # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
        output = gen_deconv(rectified, out_channels)
        output = utils.batchnorm(output)

        if dropout > 0.0:
            output = fluid.layers.dropout(output, dropout_prob=dropout)

        layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    input = fluid.layers.concat(input=[layers[-1], layers[0]], axis=1)
    rectified = fluid.layers.relu(input)
    output = gen_deconv(rectified, generator_outputs_channels)
    output = fluid.layers.tanh(output)
    layers.append(output)

    return layers[-1]


def create_model():
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, in_channels, height, width] => [batch, in_channels * 2, height, width]
        input = fluid.layers.concat(input=[discrim_inputs, discrim_targets], axis=1)

        # layer_1: [batch, in_channels * 2, 256, 256] => [batch, ndf, 128, 128]
        convolved = discrim_conv(input, a.ndf, stride=2)
        rectified = utils.lrelu(convolved, 0.2)
        layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            out_channels = a.ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride=stride)
            normalized = utils.batchnorm(convolved)
            rectified = utils.lrelu(normalized, 0.2)
            layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = fluid.layers.sigmoid(convolved)
        layers.append(output)

        return layers[-1]

    def discrim_train_program():
        inputs = fluid.layers.data(name='input_images', shape=[3, CROP_SIZE, CROP_SIZE], dtype='float32')
        targets = fluid.layers.data(name='target_images', shape=[3, CROP_SIZE, CROP_SIZE], dtype='float32')

        out_channels = 3  # int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_real = create_discriminator(inputs, targets)

        # with tf.variable_scope("discriminator", reuse=True):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_fake = create_discriminator(inputs, outputs)

        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = fluid.layers.reduce_mean(
            fluid.layers.sum(
                fluid.layers.scale(
                    x=fluid.layers.log(predict_real + EPS),
                    scale=-1.0),
                fluid.layers.log(1 - predict_fake + EPS)
            )
        )
        return [discrim_loss]

    def discrim_optimizer_program():
        return fluid.optimizer.AdamOptimizer(learning_rate=a.lr, beta1=a.beta1)

    use_cuda = a.use_cuda
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    checkpoint_config = fluid.CheckpointConfig("./checkpoints")
    discrim_trainer = fluid.Trainer(train_func=discrim_train_program, place=place,
                                    optimizer_func=discrim_optimizer_program, checkpoint_config=checkpoint_config)

    def gen_train_program():
        inputs = fluid.layers.data(name='input_images', shape=[3, CROP_SIZE, CROP_SIZE], dtype='float32')
        targets = fluid.layers.data(name='target_images', shape=[3, CROP_SIZE, CROP_SIZE], dtype='float32')

        out_channels = 3
        outputs = create_generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_real = create_discriminator(inputs, targets)

        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_fake = create_discriminator(inputs, outputs)

        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = fluid.layers.reduce_mean(
            fluid.layers.scale(
                x=fluid.layers.log(predict_fake + EPS),
                scale=-1.0
            )
        )
        gen_loss_L1 = fluid.layers.reduce_mean(
            fluid.layers.abs(targets - outputs))
        gen_loss = fluid.layers.scale(x=gen_loss_GAN, scale=a.gan_weight) + fluid.layers.scale(x=gen_loss_L1,
                                                                                               scale=a.l1_weight)
        return [gen_loss]

    def gen_optimizer_program():
        return fluid.optimizer.AdamOptimizer(learning_rate=a.lr, beta1=a.beta1)

    gen_trainer = fluid.Trainer(train_func=gen_train_program, place=place, optimizer_func=gen_optimizer_program,
                                checkpoint_config=checkpoint_config)

    # TODO: https://github.com/PaddlePaddle/Paddle/issues/7785
    # ExponentialMovingAverage()

    return Model(
        discrim_trainer=discrim_trainer,
        gen_trainer=gen_trainer,
    )


def load_examples_meta():
    data_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))

    if len(data_paths) == 0:
        raise Exception("input_dir contains no image files")
    else:
        len_data_paths = len(data_paths)

    steps_per_epoch = int(math.ceil(len_data_paths / a.batch_size))

    return Examples_Meta(
        count=len_data_paths,
        steps_per_epoch=steps_per_epoch,
    )


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples_meta = load_examples_meta()

    print(examples_meta.steps_per_epoch)
    print(examples_meta.count)

    # inputs and targets are [batch_size, channels, height, width]
    model = create_model()

    # # TODO: https://github.com/PaddlePaddle/Paddle/issues/10376
    max_steps = 2 ** 32
    if a.max_epochs is not None:
        max_steps = examples_meta.steps_per_epoch * a.max_epochs
    if a.max_steps is not None:
        max_steps = a.max_steps

    print("The max steps is: ", max_steps)

    if a.mode == "test":
        # testing
        # at most, process the test data once
        start = time.time()
        max_steps = min(examples_meta.steps_per_epoch, max_steps)
        for step in range(max_steps):
            use_cuda = a.use_cuda
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

            inferencer = fluid.Inferencer(infer_func=create_generator, param_path=a.params_dirname, place=place)

            def load_a_image(file):
                im = Image.open(file).convert('L')
                im = np.array(im)
                im = simple_transform(im=im, resize_size=a.scale_size, crop_size=CROP_SIZE, is_train=False)
                return im

            cur_dir = os.path.dirname(os.path.realpath(__file__))
            img = load_a_image(cur_dir + "/infer_img.jpg")

            results = inferencer.infer({"img": img})
            filesets = utils.save_images(results)
            for i, f in enumerate(filesets):
                print("evaluated image", f["name"])
            index_path = utils.append_index(filesets)
        print("wrote index at", index_path)
        print("rate", (time.time() - start) / max_steps)
    else:
        # training
        start = time.time()
        lists = []

        # TODO: save a image in paddlepaddle
        def event_handler(event):
            if isinstance(event, fluid.EndStepEvent):
                print(len(event.metrics))
                print("The event step is: ", event.step)
                print("The event epoch is: ", event.epoch)
                print("The time used: ", time.time() - start)
                if event.step % 100 == 0:
                    print(("Pass %d, Batch %d, Cost %f" % (event.step, event.epoch, event.metrics[0])))

            if isinstance(event, fluid.EndEpochEvent):
                avg_cost = model.gen_trainer.test(reader=facades.test_reader,
                                                  feed_order=['input_images', 'target_images'])

                avg_cost_mean = np.array(avg_cost).mean()
                print("Test with Epoch %d, avg_cost: %s" % (event.epoch, avg_cost_mean))

                # save parameters
                model.gen_trainer.save_params(a.params_dirname)
                model.discrim_trainer.save_params(a.params_dirname)
                lists.append((event.epoch, avg_cost))

                if float(avg_cost_mean) < 0.00001:  # Change this number to adjust accuracy
                    model.gen_trainer.stop()
                    model.discrim_trainer.stop()
                elif math.isnan(float(avg_cost_mean)):
                    sys.exit("got NaN loss, training failed.")

        model.discrim_trainer.train(num_epochs=1, event_handler=event_handler, reader=facades.train_reader(
            resize_size=a.scale_size, crop_size=CROP_SIZE, batch_size=a.batch_size,
            lab_colorization=a.lab_colorization, which_direction=a.which_direction
        ),
                                        feed_order=['input_images', 'target_images'])
        model.gen_trainer.train(num_epochs=1, event_handler=event_handler, reader=facades.train_reader(
            resize_size=a.scale_size, crop_size=CROP_SIZE, batch_size=a.batch_size,
            lab_colorization=a.lab_colorization, which_direction=a.which_direction
        ),
                                    feed_order=['input_images', 'target_images'])

        print("time duration: ", (time.time() - start))
        print("rate: ", (time.time() - start) / max_steps)
        best = sorted(lists, key=lambda list: float(list[1]))[0]
        print("Best pass is %s, testing avgcost is %s" % (best[0], best[1]))


main()
