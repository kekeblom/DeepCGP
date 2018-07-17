import gpflow
import numpy as np
import tensorflow as tf
from gpflow.decors import params_as_tensors

class Conv2dMean(gpflow.mean_functions.MeanFunction):
    def __init__(self, filter_size, feature_maps_in, feature_maps_out=1, stride=1):
        super().__init__()
        self.filter_size = filter_size
        self.feature_maps_in = feature_maps_in
        self.feature_maps_out = feature_maps_out
        self.stride = stride
        self.conv_filter = gpflow.Param(self._init_filter())

    @params_as_tensors
    def __call__(self, NHWC_X):
        convolved = tf.nn.conv2d(NHWC_X, self.conv_filter,
                strides=[1, self.stride, self.stride, 1],
                padding="VALID",
                data_format="NHWC")
        NHWC = tf.shape(NHWC_X)
        return tf.reshape(convolved, [NHWC[0], -1])

    def _init_filter(self):
        # Only supports square filters with odd size for now.
        identity_filter = np.zeros((self.filter_size, self.filter_size), dtype=gpflow.settings.float_type)
        identity_filter[self.filter_size // 2, self.filter_size // 2] = 1.0
        return np.tile(identity_filter[:, :, None, None], [1, 1, self.feature_maps_in, self.feature_maps_out])

class PatchwiseConv2d(Conv2dMean):
    def __init__(self, filter_size, feature_maps_in, out_height, out_width):
        super().__init__(filter_size, feature_maps_in)
        self.out_height = out_height
        self.out_width = out_width

    @params_as_tensors
    def __call__(self, PNL_patches):
        kernel = tf.reshape(self.conv_filter, [self.filter_size**2 * self.feature_maps_in,
            self.feature_maps_in])
        def foreach_patch(NL_patches):
            return tf.matmul(NL_patches, kernel)
        PN1_out = tf.map_fn(foreach_patch, PNL_patches)
        PNL = tf.shape(PNL_patches)
        return tf.reshape(tf.transpose(PN1_out, [2, 1, 0]), [PNL[1], PNL[0]])


