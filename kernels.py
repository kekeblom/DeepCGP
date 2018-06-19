import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow.features import InducingFeature
from gpflow.params import Parameter
from sklearn import cluster

def _sample(tensor, count):
    chosen_indices = np.random.choice(np.arange(tensor.shape[0]), count)
    return tensor[chosen_indices]

class ConvKernel(gpflow.kernels.Kernel):
    # Loosely based on https://github.com/markvdw/convgp/blob/master/convgp/convkernels.py
    def __init__(self, base_kernel, image_size, window_size=3, stride=1, channels=1):
        super().__init__(input_dim=window_size*window_size)
        self.base_kernel = base_kernel
        self.image_size = image_size
        self.window_size = window_size
        self.stride = stride
        self.dilation = 1
        self.patch_size = (window_size, window_size)
        self.color_channels = channels

    def _get_patches(self, X):
        """_get_patches

        :param X: N x height x width x color_channels
        :returns N x _patch_count x _patch_length
        """
        # X: batch x height x width x channels
        # returns: batch x height x width x channels * patches
        patches = tf.extract_image_patches(X,
                [1, self.window_size, self.window_size, 1],
                [1, self.stride, self.stride, 1],
                [1, self.dilation, self.dilation, 1],
                "VALID")
        N = tf.shape(X)[0]
        return tf.reshape(patches, (N, self._patch_count, self._patch_length))

    def K(self, X, X2=None):
        patch_length = self._patch_length
        patches = tf.reshape(self._get_patches(X), [-1, patch_length])

        if X2 is None:
            patches2 = patches
        else:
            patches2 = tf.reshape(self._get_patches(X2), [-1, patch_length])

        # K: batch * patches x batch * patches
        K = self.base_kernel.K(patches, patches2)
        X_shape = tf.shape(X)
        reshaped_K  = tf.reshape(K, (X_shape[0], self._patch_count, -1, self._patch_count))
        # Sum over the patches.
        K = tf.reduce_sum(reshaped_K, [1, 3]) / (self._patch_count ** 2)
        return K

    def Kdiag(self, X):
        # patches: N x _patch_count x _patch_length
        # Compute auto correlation in the patch space.
        patches = self._get_patches(X)

        def sumK(p):
            return tf.reduce_sum(self.base_kernel.K(p))
        return tf.map_fn(sumK, patches) / (self._patch_count ** 2)

    def Kzx(self, Z, X):
        # Patches: N x _patch_count x _patch_length
        patches = self._get_patches(X)
        patches = tf.reshape(patches, (-1, self._patch_length))
        # Kzx shape: M x N * _patch_count
        Kzx = self.base_kernel.K(Z, patches)
        M = tf.shape(Z)[0]
        N = tf.shape(X)[0]
        # Reshape to M x N x _patch_count then sum over patches.
        Kzx = tf.reshape(Kzx, (M, N, self._patch_count))
        Kzx = tf.reduce_sum(Kzx, [2])
        return Kzx / self._patch_count

    def Kzz(self, Z):
        return self.base_kernel.K(Z)

    def init_inducing_patches(self, X, M):
        # Randomly sample images and patches.
        sample_size = 125
        random_sample = _sample(X, sample_size)
        patches = self.autoflow_patches(random_sample)
        patches = patches.reshape(sample_size * self._patch_count, self._patch_length)

        k_means = cluster.KMeans(n_clusters=M,
                init='random', n_jobs=-1)
        k_means.fit(patches)
        return k_means.cluster_centers_.reshape(M, self._patch_length)

    @property
    def _patch_length(self):
        """_patch_length: the number of elements in a patch."""
        return self.color_channels * np.prod(self.patch_size)

    @property
    def _patch_count(self):
        """_patch_count: the amount of patches in one image."""
        return (self.image_size[0] - self.patch_size[0] + 1) * (
                self.image_size[1] - self.patch_size[1] + 1) * self.color_channels

    @gpflow.autoflow((settings.float_type,))
    def autoflow_patches(self, X):
        return self._get_patches(X)


class PatchInducingFeature(InducingFeature):
    def __init__(self, Z):
        super().__init__()
        self.Z = Parameter(Z, dtype=settings.float_type)

    def __len__(self):
        return self.Z.shape[0]

    def Kuu(self, kern, jitter=0.0):
        return kern.Kzz(self.Z) + tf.eye(len(self), dtype=settings.dtypes.float_type) * jitter

    def Kuf(self, kern, Xnew):
        return kern.Kzx(self.Z, Xnew)

gpflow.features.conditional.register(PatchInducingFeature,
        gpflow.features.default_feature_conditional)

