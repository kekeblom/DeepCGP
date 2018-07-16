import functools
import numpy as np
import tensorflow as tf

class View(object):
    """Views construct the patches and define the number of outputs the
    conv layer will have.
    A view should implement extract_patches_PNL and define patch_count and patch_length.
    """
    def _extract_patches_PNL(self, *args):
        raise NotImplementedError()

    def mean_view(self, NHWC_X, PNL_patches):
        """Returns a N x H x W x C shaped tensor. The view of the data passed to the mean function."""
        return NHWC_X

class FullView(View):
    """The full view uses all patches of the image."""
    def __init__(self, input_size, filter_size, feature_maps):
        self.input_size = input_size
        self.stride = 1
        self.dilation = 1
        self.filter_size = filter_size
        self.feature_maps = feature_maps
        self.patch_shape = (filter_size, filter_size)
        self.patch_count = self._patch_count()
        self.patch_length = self._patch_length()

    def _extract_image_patches(self, NHWC_X):
        # returns: N x H x W x C * P
        return tf.extract_image_patches(NHWC_X,
                [1, self.filter_size, self.filter_size, 1],
                [1, self.stride, self.stride, 1],
                [1, self.dilation, self.dilation, 1],
                "VALID")

    def extract_patches_PNL(self, NHWC_X):
        NHWK_patches = self._extract_image_patches(NHWC_X)
        KNHW_patches = tf.transpose(NHWK_patches, [3, 0, 1, 2])
        # Currently only one filter map is supported. This should blow up
        # if this is not the case.
        N = tf.shape(NHWC_X)[0]
        return tf.reshape(KNHW_patches, [self.patch_count, N, self.patch_length])

    def extract_patches(self, NHWC_X):
        """extract_patches

        :param X: N x height x width x feature_maps
        :returns N x patch_count * feature_maps x patch_length
        """
        # X: batch x height * width * feature_maps
        NHWK_patches = self._extract_image_patches(NHWC_X)
        N = tf.shape(NHWC_X)[0]
        NKHW_patches = tf.transpose(NHWK_patches, [0, 3, 1, 2])
        return tf.reshape(NKHW_patches, [N, self.patch_count * self.feature_maps, self.patch_length])

    def _patch_length(self):
        """The number of elements in a patch."""
        return self.feature_maps * np.prod(self.patch_shape)

    def _patch_count(self):
        """The amount of patches in one image."""
        return (self.input_size[0] - self.patch_shape[0] + 1) * (
                self.input_size[1] - self.patch_shape[1] + 1) * self.feature_maps

class RandomPartialView(View):
    def __init__(self, input_size, filter_size, feature_maps,
            patch_count):
        self.input_size = input_size
        self.stride = 1
        self.dilation = 1
        self.feature_maps = feature_maps
        self.filter_size = filter_size
        self.patch_shape = (filter_size, filter_size)
        self.patch_count = patch_count
        self.patch_length = self._patch_length()
        self.patch_indices = self._select_patches()
        self.out_image_height, self.out_image_width = self._out_image_size()

    def _select_patches(self):
        ys = np.arange(0, self.input_size[0] - self.filter_size)
        xs = np.arange(0, self.input_size[1] - self.filter_size)
        taken = {}
        patches = []
        while len(patches) < self.patch_count:
            y = np.random.choice(ys)
            x = np.random.choice(xs)
            if (y, x) in taken:
                continue
            else:
                taken[(y, x)] = True
                y = slice(y, y + self.filter_size)
                x = slice(x, x + self.filter_size)
                patches.append([y, x])
        large_int = 10**5
        def ordering(yx):
            return yx[0].start * large_int + yx[1].start
        patches.sort(key=ordering)
        return patches

    def extract_patches_PNL(self, NHWC_X):
        NHWC = tf.shape(NHWC_X)
        NL = [NHWC[0], self.patch_length]
        tensors = []
        for yx in self.patch_indices:
            patches = NHWC_X[:, yx[0], yx[1], :]
            tensors.append(tf.reshape(patches, NL))
        PNL_stacked = tf.stack(tensors)
        return PNL_stacked

    def mean_view(self, _, PNL_patches):
        return PNL_patches

    def _patch_length(self):
        return np.prod(self.patch_shape) * self.feature_maps

    def _out_image_size(self):
        side = int(np.sqrt(self.patch_count))
        return (side, side)



