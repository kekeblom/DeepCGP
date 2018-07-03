import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, features, conditionals, transforms
from doubly_stochastic_dgp.layers import Layer
from kernels import ConvKernel, PatchInducingFeature

class ConvLayer(Layer):
    def __init__(self, base_kernel, num_outputs, mean_function, feature=None,
            input_size=None,
            feature_maps=None,
            filter_size=None,
            white=False,
            **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.input_size = input_size
        self.filter_size = filter_size
        self.stride = 1
        self.dilation = 1
        self.patch_shape = (filter_size, filter_size)
        self.feature_maps = feature_maps
        self.patch_count = self._patch_count()
        self.patch_length = self._patch_length()
        self.patch_weights = gpflow.Param(np.ones(self.patch_count, dtype=settings.float_type))

        self.white = white

        self.conv_kernel = ConvKernel(base_kernel, input_size, filter_size, feature_maps=feature_maps)
        self.feature = feature

        self.num_outputs = num_outputs
        self.num_inducing = len(feature)

        MO_q_mu = np.zeros((self.num_inducing, num_outputs), dtype=settings.float_type)
        self.MO_q_mu = gpflow.Param(MO_q_mu)

        #TODO figure out if we need whitened vs non-whitened GP.
        if not self.white:
            self.OMM_q_sqrt = self._init_q_sqrt_from_Z()
        else:
            self.OMM_q_sqrt = gpflow.Param(np.tile(
                    np.eye(self.num_inducing, dtype=settings.float_type),
                    [num_outputs, 1, 1]))

        self.mean_function = mean_function

        self._build_cholesky()

    def conditional_ND(self, N_X, full_cov=False):
        """conditional_ND Returns the mean and variance of the normal distribution
        corresponding to p(y | X, Z).

        dimension O: D out

        :param ON_X: The input X of shape O x N
        :param full_cov: if true, var is in (N, N, D_out) instead of (N, D_out) i.e. we
        also compute entries outside the diagonal.
        """
        self.build_cholesky_if_needed()

        NM_Knm = conditionals.Kuf(self.feature, self.kern, N_X)

        # A = NM_Knm @ MM_Kmm_inv
        MN_A = tf.matrix_triangular_solve(self.MM_Lu, NM_Knm, lower=True)
        if not self.white:
            # A = A @ Lu^-T
            MN_A = tf.matrix_triangular_solve(tf.transpose(self.Lu), MN_A, lower=False)

        NO_mean = tf.matmul(MN_A, self.MO_q_mu, transpose_a=True)

        if self.white:
            OMM_SK = -tf.eye(self.num_inducing, dtype=settings.float_type)[None, :, :]
        else:
            OMM_SK = -self.OMM_Ku

        OMM_SK = OMM_SK + tf.matmul(self.OMM_q_sqrt, self.OMM_q_sqrt, transpose_b=True)

        OMN_A = tf.tile(MN_A[None, :, :], [self.num_outputs, 1, 1])
        OMN_B = OMM_SK @ OMN_A

        if full_cov:
            ONN_Knn = self.kern.K(N_X)
            ONN_var = ONN_Knn + tf.matmul(OMN_A, OMN_B, transpose_a=True)
            var = tf.transpose(ONN_var, [1, 2, 0])
        else:
            ON_Kdiag = self.kern.Kdiag(N_X)
            OMN_delta_cov = OMN_A * OMN_B
            ON_delta_cov = tf.reduce_sum(OMN_delta_cov, 1)
            ON_var = ON_Kdiag + ON_delta_cov

            var = tf.transpose(ON_var, [1, 0])

        return NO_mean + self.mean_function(N_X), var

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior

        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        KL = -0.5 * self.num_outputs * self.num_inducing
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.OMM_q_sqrt) ** 2))

        if not self.white:
            KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.MM_Lu))) * self.num_outputs
            KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.OMM_Lu, self.OMM_q_sqrt, lower=True)))
            MM_Kinv_m = tf.cholesky_solve(self.MM_Lu, self.MO_q_mu)
            KL += 0.5 * tf.reduce_sum(self.MO_q_mu * MM_Kinv_m)
        else:
            KL += 0.5 * tf.reduce_sum(tf.square(self.OMM_q_sqrt))
            KL += 0.5 * tf.reduce_sum(self.MO_q_mu**2)

        return KL

    def _get_image_patches(self, NCHW_X):
        """Extracts patches from an image.

        :param X: N x height x width x color_channels
        :returns N x _patch_count x patch_length
        """
        patches = tf.extract_image_patches(NCHW_X,
                [1, self.filter_size, self.filter_size, 1],
                [1, self.stride, self.stride, 1],
                [1, self.dilation, self.dilation, 1],
                "VALID")
        N = tf.shape(NCHW_X)[0]
        return tf.reshape(patches, (N, self.patch_count, self.patch_length))

    def _init_q_sqrt_from_Z(self):
        MM_Ku = conditionals.Kuu(self.feature, self.conv_kernel, jitter=settings.jitter)
        MM_Lu = tf.linalg.choleksy(Ku)
        MM_Lu = self.enquire_session().run(MM_Lu)
        return gpflow.Param(np.tile(MM_Lu[None, :, :], [self.num_outputs, 1, 1]))

    def _build_cholesky(self):
        self.MM_Ku = conditionals.Kuu(self.feature, self.conv_kernel, jitter=settings.jitter)
        self.MM_Lu = tf.linalg.cholesky(self.Ku)
        self.OMM_Lu = tf.tile(Lu[None, :, :], [self.num_outputs, 1, 1])
        self.OMM_Ku = tf.tile(Ku[None, :, :], [self.num_outputs, 1, 1])

    def _patch_length(self):
        """The number of elements in a patch."""
        return self.feature_maps * np.prod(self.patch_shape)

    def _patch_count(self):
        """The amount of patches in one image."""
        return (self.input_size[0] - self.patch_shape[0] + 1) * (
                self.input_size[1] - self.patch_shape[1] + 1) * self.feature_maps


