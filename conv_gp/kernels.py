import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow.features import InducingPointsBase
from gpflow.multioutput.features import SeparateIndependentMof
from gpflow.params import Parameter
from gpflow.dispatch import dispatch
from sklearn import cluster

def _sample(tensor, count):
    chosen_indices = np.random.choice(np.arange(tensor.shape[0]), count)
    return tensor[chosen_indices]

class ConvKernel(gpflow.kernels.Kernel):
    # Loosely based on https://github.com/markvdw/convgp/blob/master/convgp/convkernels.py
    def __init__(self, base_kernel, view):
        super().__init__(input_dim=np.prod(view.input_size))
        self.base_kernel = base_kernel
        self.view = view
        self.patch_length = view.patch_length
        self.patch_count = view.patch_count
        self.patch_weights = gpflow.Param(np.ones(self.patch_count, dtype=settings.float_type))
        self.image_size = self.view.input_size

    def _reshape_X(self, ND_X):
        ND = tf.shape(ND_X)
        return tf.reshape(ND_X, [ND[0]] + list(self.view.input_size))

    def K(self, ND_X, X2=None):
        NHWC_X = self._reshape_X(ND_X)
        patch_length = self.patch_length
        # N * P x L
        patches = tf.reshape(self.view.extract_patches(NHWC_X), [-1, patch_length])

        if X2 is None:
            patches2 = patches
        else:
            patches2 = tf.reshape(self.view.extract_patches(X2), [-1, patch_length])

        # K: batch * patches x batch * patches
        K = self.base_kernel.K(patches, patches2)
        X_shape = tf.shape(NHWC_X)
        # Reshape to batch x patch_count x batch x patch_count
        K = tf.reshape(K, (X_shape[0], self.patch_count, X_shape[0], self.patch_count))

        w = self.patch_weights
        W = w[None, :] * w[:, None] # P x P
        W = W[None, :, None, :] # 1 x P x 1 x P
        K = K * W

        # Sum over the patch dimensions.
        return tf.reduce_sum(K, [1, 3]) / (self.patch_count ** 2)

    def Kdiag(self, ND_X):
        NHWC_X = self._reshape_X(ND_X)
        # Compute auto correlation in the patch space.
        # patches: N x patch_count x patch_length
        patches = self.view.extract_patches(NHWC_X)
        w = self.patch_weights
        W = w[None, :] * w[:, None]
        def sumK(p):
            return tf.reduce_sum(self.base_kernel.K(p) * W)
        return tf.map_fn(sumK, patches) / (self.patch_count ** 2)

    def Kzx(self, Z, ND_X):
        NHWC_X = self._reshape_X(ND_X)
        # Patches: N x patch_count x patch_length
        patches = self.view.extract_patches(NHWC_X)
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

def _cluster_patches(NHWC_X, M, patch_size):
        NHWC = NHWC_X.shape
        patch_length = patch_size ** 2 * NHWC[3]
        # Randomly sample images and patches.
        patches = np.zeros((M, patch_length), dtype=settings.float_type)
        patches_per_image = 1
        samples_per_inducing_point = 100
        for i in range(M * samples_per_inducing_point // patches_per_image):
            # Sample a random image, compute the patches and sample some random patches.
            image = _sample(NHWC_X, 1)[0]
            sampled_patches = _sample_patches(image, patches_per_image,
                    patch_size, patch_length)
            patches[i*patches_per_image:(i+1)*patches_per_image] = sampled_patches

        k_means = cluster.KMeans(n_clusters=M,
                init='random', n_jobs=-1)
        k_means.fit(patches)
        return k_means.cluster_centers_

class PatchInducingFeatures(InducingPointsBase):
    @classmethod
    def from_images(cls, NHWC_X, M, patch_size):
        features = _cluster_patches(NHWC_X, M, patch_size)
        return PatchInducingFeatures(features)

class IndependentPatchInducingFeatures(SeparateIndependentMof):
    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, count):
        """Inducing points will be of shape count x M x patch_length"""
        patch_length = patch_size ** 2 * NHWC_X.shape[3]
        patches = _cluster_patches(NHWC_X, M, patch_size)
        np.random.shuffle(patches) # Shuffle along M dimension.
        inducing_points = patches.reshape(count, M // count, patch_length)
        return IndependentPatchInducingFeatures([p for p in inducing_points])


@dispatch(PatchInducingFeatures, ConvKernel)
def Kuu(feature, kern, jitter=0.0):
    return kern.Kzz(feature.Z) + tf.eye(len(feature), dtype=settings.dtypes.float_type) * jitter

@dispatch(PatchInducingFeatures, ConvKernel, object)
def Kuf(feature, kern, Xnew):
    return kern.Kzx(feature.Z, Xnew)



