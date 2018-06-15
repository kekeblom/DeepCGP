import numpy as np
import tensorflow as tf
import gpflow
import observations
from sklearn import cluster

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
        # X: batch x height x width x channels
        # returns: batch x height x width x channels * patches
        return tf.extract_image_patches(X,
                [1, self.window_size, self.window_size, 1],
                [1, self.stride, self.stride, 1],
                [1, self.dilation, self.dilation, 1],
                "VALID")

    def K(self, X, X2=None):
        patch_length = self.color_channels * np.prod(self.patch_size)
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
        patches = self._get_patches(X)
        def sumK(p):
            return tf.reduce_sum(self.base_kernel.K(p))
        return tf.map_fn(sumK, patches) / (self._patch_count ** 2)

    @property
    def _patch_count(self):
        return (self.image_size[0] - self.patch_size[0] + 1) * (
                self.image_size[1] - self.patch_size[1] + 1) * self.color_channels

class Classification(object):
    def __init__(self):
        self._load_data()
        self.num_inducing = 25
        Z = self._compute_Z()
        window_size = 3
        patch_size = window_size**2
        kernel = ConvKernel(
                gpflow.kernels.RBF(input_dim=patch_size),
                image_size=(28, 28),
                window_size=window_size,
                stride=1,
                channels=1)
        self.model = gpflow.models.SVGP(self.X_train, self.Y_train,
                kern=kernel,
                Z=Z,
                num_latent=10,
                likelihood=gpflow.likelihoods.MultiClass(10),
                whiten=True)


    def run(self):
        optimizer = gpflow.train.GradientDescentOptimizer(learning_rate=0.01)
        optimizer.minimize(self.model)

    def _compute_Z(self):
        X = self.X_train.reshape(-1, 28**2)
        k_means = cluster.KMeans(n_clusters=self.num_inducing,
                init='random', n_jobs=-1)
        k_means.fit(X)
        print("z initialized")
        return k_means.cluster_centers_.reshape(-1, 28, 28, 1)

    def _load_data(self):
        (self.X_train, self.Y_train), (
                self.X_test, self.Y_test) = observations.mnist('/tmp/mnist/')
        def reshape(X):
            return X.reshape(-1, 28, 28, 1)
        self.X_train = reshape(self.X_train)[0:10000].astype(gpflow.settings.float_type)
        self.X_test = reshape(self.X_test).astype(gpflow.settings.float_type)


if __name__ == "__main__":
    experiment = Classification()
    experiment.run()

