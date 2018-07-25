import os
import paddle.fluid as fluid
import numpy as np


def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def deprocess(image):
    # [-1, 1] => [0, 1]
    return (image + 1) / 2


def preprocess_lab(lab):
    # lab [w, h, c]
    # L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
    lab1 = np.swapaxes(lab, 0, 2)
    lab2 = np.swapaxes(lab1, 1, 2)
    L_chan, a_chan, b_chan = np.vsplit(lab2, 2)

    # L_chan: black and white with input range [0, 100]
    # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
    # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
    return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
    return np.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    """
    >> > print (session.run(y))
    [[[[0  1]
       [2  3]]

     [[4  5]
        [6  7]]]


    [[[8  9]
      [10 11]]

    [[12 13]
    [14
    15]]]]
    >> > print (session.run(u3))
    [array([[[0, 2],
             [4, 6]],

            [[8, 10],
             [12, 14]]], dtype=int32),
    array([[[1, 3],
            [5, 7]],

            [[9, 11],
            [13, 15]]], dtype=int32)]
    """

    # a_chan, b_chan = tf.unstack(image, axis=3)
    image1 = np.swapaxes(image, 1, 3)
    image2 = np.swapaxes(image1, 2, 3)
    image3 = np.swapaxes(image2, 0, 1)
    a_chan, b_chan = np.vsplit(image3, 2)

    L_chan = np.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def lrelu(x, a):
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2

    # this block looks like it has 2 inputs on the graph unless we do this
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * fluid.layers.abs(x)


def batchnorm(inputs):
    return fluid.layers.batch_norm(input=inputs, epsilon=1e-5, momentum=0.1)


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    srgb_pixels = srgb.reshape((-1, 3))

    # linear_mask = fluid.layers.cast(srgb_pixels <= 0.04045, dtype=np.float32)
    # exponential_mask = fluid.layers.cast(srgb_pixels > 0.04045, dtype=np.float32)
    linear_mask_lambda = lambda pixel: np.float32(1.0) if pixel <= 0.04045 else np.float32(0.0)
    vfunc_linear_mask = np.vectorize(linear_mask_lambda, otypes=[np.float32])
    linear_mask = vfunc_linear_mask(srgb_pixels)

    exponential_mask_lambda = lambda pixel: np.float32(1.0) if pixel > 0.04045 else np.float32(0.0)
    vfunc_exponential_mask = np.vectorize(exponential_mask_lambda, otypes=[np.float32])
    exponential_mask = vfunc_exponential_mask(srgb_pixels)

    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
            ((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    rgb_to_xyz = np.array([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ])
    xyz_pixels = np.multiply(rgb_pixels, rgb_to_xyz)

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

    # normalize for D65 white point
    xyz_normalized_pixels = np.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

    epsilon = 6 / 29
    # linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
    # exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)

    linear_mask_lambda = lambda pixel: np.float32(1.0) if pixel <= (epsilon ** 3) else np.float32(0.0)
    vfunc_linear_mask = np.vectorize(linear_mask_lambda, otypes=[np.float32])
    linear_mask = vfunc_linear_mask(srgb_pixels)

    exponential_mask_lambda = lambda pixel: np.float32(1.0) if pixel > (epsilon ** 3) else np.float32(0.0)
    vfunc_exponential_mask = np.vectorize(exponential_mask_lambda, otypes=[np.float32])
    exponential_mask = vfunc_exponential_mask(srgb_pixels)

    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
            xyz_normalized_pixels ** (1 / 3)) * exponential_mask

    # convert to lab
    fxfyfz_to_lab = np.array([
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ])
    lab_pixels = np.multiply(fxfyfz_pixels, fxfyfz_to_lab) + np.array([-16.0, 0.0, 0.0])

    return lab_pixels.reshape(srgb.shape)


def lab_to_rgb(lab):
    lab_pixels = lab.reshape((-1, 3))

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    # convert to fxfyfz
    lab_to_fxfyfz = np.array([
        #   fx      fy        fz
        [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
        [1 / 500.0, 0.0, 0.0],  # a
        [0.0, 0.0, -1 / 200.0],  # b
    ])
    fxfyfz_pixels = np.multiply(lab_pixels + np.array([16.0, 0.0, 0.0]), lab_to_fxfyfz)

    # convert to xyz
    epsilon = 6 / 29
    # linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
    # exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)

    linear_mask_lambda = lambda pixel: np.float32(1.0) if pixel <= epsilon else np.float32(0.0)
    vfunc_linear_mask = np.vectorize(linear_mask_lambda, otypes=[np.float32])
    linear_mask = vfunc_linear_mask(fxfyfz_pixels)

    exponential_mask_lambda = lambda pixel: np.float32(1.0) if pixel > epsilon else np.float32(0.0)
    vfunc_exponential_mask = np.vectorize(exponential_mask_lambda, otypes=[np.float32])
    exponential_mask = vfunc_exponential_mask(fxfyfz_pixels)

    xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (
            fxfyfz_pixels ** 3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = np.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

    xyz_to_rgb = np.array([
        #     r           g          b
        [3.2404542, -0.9692660, 0.0556434],  # x
        [-1.5371385, 1.8760108, -0.2040259],  # y
        [-0.4985314, 0.0415560, 1.0572252],  # z
    ])
    rgb_pixels = np.multiply(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    rgb_pixels = np.clip(rgb_pixels, 0.0, 1.0)
    # linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
    # exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)

    linear_mask_lambda = lambda pixel: np.float32(1.0) if pixel <= 0.0031308 else np.float32(0.0)
    vfunc_linear_mask = np.vectorize(linear_mask_lambda, otypes=[np.float32])
    linear_mask = vfunc_linear_mask(rgb_pixels)

    exponential_mask_lambda = lambda pixel: np.float32(1.0) if pixel > 0.0031308 else np.float32(0.0)
    vfunc_exponential_mask = np.vectorize(exponential_mask_lambda, otypes=[np.float32])
    exponential_mask = vfunc_exponential_mask(rgb_pixels)

    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
            (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

    return srgb_pixels.reshape(lab.shape)


def save_images(a, fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(a, filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path
