import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow.features import InducingPointsBase
from gpflow.params import Parameter
from gpflow.dispatch import dispatch
from sklearn import cluster

def _sample(tensor, count):
    chosen_indices = np.random.choice(np.arange(tensor.shape[0]), count)
    return tensor[chosen_indices]

class PatchMixin(object):
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


class ConvKernel(gpflow.kernels.Kernel, PatchMixin):
    # Loosely based on https://github.com/markvdw/convgp/blob/master/convgp/convkernels.py
    def __init__(self, base_kernel, image_size, filter_size=5, stride=1, feature_maps=1):
        super().__init__(input_dim=filter_size*filter_size)
        self.base_kernel = base_kernel
        self.image_size = image_size
        self.filter_size = filter_size
        self.stride = stride
        self.dilation = 1
        self.patch_size = (filter_size, filter_size)
        self.feature_maps = feature_maps
        self.patch_weights = gpflow.Param(np.ones(self.patch_count, dtype=settings.float_type))
        self.patch_length = self._patch_length()
        self.patch_count = self._patch_count()

    def _reshape_X(self, X):
        """
        Reshapes the input from N x D to N x image height x image width x feature_maps

        :param X: Tensorflow tensor of size N x D
        :returns: Tensorflow tensor of size N x height x width x feature_maps
        """
        X_shape = tf.shape(X)
        return tf.reshape(X, (X_shape[0], *self.image_size, self.feature_maps))

    def K(self, NHWC_X, X2=None):
        patch_length = self.patch_length
        patches = tf.reshape(self.extract_patches(NHWC_X), [-1, patch_length])

        if X2 is None:
            patches2 = patches
        else:
            patches2 = tf.reshape(self.extract_patches(X2), [-1, patch_length])

        # K: batch * patches x batch * patches
        K = self.base_kernel.K(patches, patches2)
        X_shape = tf.shape(NHWC_X)
        # Reshape to batch x patch_count x batch x patch_count
        K  = tf.reshape(K, (X_shape[0], self.patch_count, X_shape[0], self.patch_count))

        w = self.patch_weights
        W = w[None, :] * w[:, None] # P x P
        W = W[None, :, None, :] # 1 x P x 1 x P
        K = K * W

        # Sum over the patch dimensions.
        return tf.reduce_sum(K, [1, 3]) / (self.patch_count ** 2)

    def Kdiag(self, NHWC_X):
        # Compute auto correlation in the patch space.
        # patches: N x patch_count x patch_length
        patches = self.extract_patches(NHWC_X)
        w = self.patch_weights
        W = w[None, :] * w[:, None]
        def sumK(p):
            return tf.reduce_sum(self.base_kernel.K(p) * W)
        return tf.map_fn(sumK, patches) / (self.patch_count ** 2)

    def Kzx(self, Z, NHWC_X):
        # Patches: N x patch_count x patch_length
        patches = self.extract_patches(NHWC_X)
        patches = tf.reshape(patches, (-1, self.patch_length))
        # Kzx shape: M x N * patch_count
        Kzx = self.base_kernel.K(Z, patches)
        M = tf.shape(Z)[0]
        N = tf.shape(NHWC_X)[0]
        # Reshape to M x N x patch_count then sum over patches.
        Kzx = tf.reshape(Kzx, (M, N, self.patch_count))

        w = self.patch_weights
        Kzx = Kzx * w

        Kzx = tf.reduce_sum(Kzx, [2])
        return Kzx / self.patch_count

    def Kzz(self, Z):
        return self.base_kernel.K(Z)

def _sample_patches(HW_image, N, patch_size, patch_length):
    out = np.zeros((N, patch_length))
    for i in range(N):
        patch_y = np.random.randint(0, HW_image.shape[0] - patch_size)
        patch_x = np.random.randint(0, HW_image.shape[1] - patch_size)
        out[i] = HW_image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size].reshape(patch_length)
    return out

class PatchInducingFeature(InducingPointsBase):
    @classmethod
    def from_images(cls, X, M, patch_size):
        patch_length = patch_size ** 2
        # Randomly sample images and patches.
        patches = np.zeros((M, patch_length), dtype=settings.float_type)
        patches_per_image = 1
        samples_per_inducing_point = 100
        for i in range(M * samples_per_inducing_point // patches_per_image):
            # Sample a random image, compute the patches and sample some random patches.
            image = _sample(X, 1)[0]
            sampled_patches = _sample_patches(image, patches_per_image,
                    patch_size, patch_length)
            patches[i*patches_per_image:(i+1)*patches_per_image] = sampled_patches

        k_means = cluster.KMeans(n_clusters=M,
                init='random', n_jobs=-1)
        k_means.fit(patches)
        points = k_means.cluster_centers_.reshape(M, patch_length)
        return PatchInducingFeature(points)


@dispatch(PatchInducingFeature, ConvKernel)
def Kuu(feature, kern, jitter=0.0):
    return kern.Kzz(feature.Z) + tf.eye(len(feature), dtype=settings.dtypes.float_type) * jitter

@dispatch(PatchInducingFeature, ConvKernel, object)
def Kuf(feature, kern, Xnew):
    return kern.Kzx(feature.Z, Xnew)


# gpflow.features.conditional.register(PatchInducingFeature,
#         gpflow.features.default_feature_conditional)

