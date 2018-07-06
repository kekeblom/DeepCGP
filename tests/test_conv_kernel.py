import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow.test_util import GPflowTestCase
from .context import conv_gp
from conv_gp.layers import MultiOutputConvKernel
from conv_gp.kernels import PatchInducingFeature

class TestMultiOutputConvKernel(GPflowTestCase):
    def setUp(self):
        np.random.seed(0)

    def setup_kernel(self):
        self.filter_size = 3
        self.M = 16
        self.image_width = 28
        self.image_height = 28
        self.images = np.random.randn(32, self.image_height, self.image_width, 1)
        self.rbf = gpflow.kernels.RBF(self.filter_size**2)
        self.Z = np.random.randn(self.M, self.filter_size**2)
        self.conv_kernel = MultiOutputConvKernel(
            base_kernel=self.rbf,
            input_size = (self.image_width, self.image_height),
            filter_size=self.filter_size,
            feature_maps=1
        )

    def get_patches(self, images):
        P = self.conv_kernel.patch_count
        N = len(images)
        out = np.zeros((P, N, self.conv_kernel.patch_length),
                dtype=settings.float_type)
        fs = self.filter_size
        x, y = 0, 0
        i = 0
        while y+fs < self.image_height:
            while x+fs < self.image_width:
                for n in range(N):
                    out[i, n, :] = images[n, y:y+fs,
                            x:x+fs, 0].ravel()
                    i += 1
                    x, y = x+1, y+1
        return tf.constant(out)

    def test_Kuu_basic(self):
        with self.test_context() as sess:
            self.setup_kernel()
            Z = tf.constant(self.Z, dtype=settings.float_type)
            Kuu = self.conv_kernel.Kuu(Z)
            Kuu = sess.run(Kuu)
            self.assertEqual(Kuu.shape, (self.M, self.M))
            Z = sess.run(Z)

            auto_covariance = self.rbf.compute_K_symm(Z[0][None])
            self.assertTrue(auto_covariance[0, 0] - Kuu[0, 0] < settings.jitter)

    def test_Kuf_basic(self):
        with self.test_context() as sess:
            self.setup_kernel()
            X = self.images[0:2]
            patches = self.get_patches(X)
            Z = tf.constant(self.Z, dtype=settings.float_type)

            Kuf = sess.run(self.conv_kernel.Kuf(Z, patches))
            self.assertEqual(Kuf.shape[0], self.conv_kernel.patch_count)
            self.assertEqual(Kuf.shape[1], len(self.Z))
            self.assertEqual(Kuf.shape[2], len(X))


