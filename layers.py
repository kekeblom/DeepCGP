import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, features, conditionals, transforms
from gpflow.dispatch import dispatch
from doubly_stochastic_dgp.layers import Layer
from kernels import PatchMixin, ConvKernel, PatchInducingFeature


class ConvLayer(Layer, PatchMixin):
    def __init__(self, base_kernel, mean_function, feature=None,
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

        self.num_outputs = self.patch_count
        self.num_inducing = len(feature)

        M1_q_mu = np.zeros((self.num_inducing, 1), dtype=settings.float_type)
        self.M1_q_mu = gpflow.Param(M1_q_mu)

        #TODO figure out if we need whitened vs non-whitened GP.
        if not self.white:
            #TODO this should be size MM and not individually learned for each output.
            self.OMM_q_s = self._init_q_s_from_Z()
        else:
            self.OMM_q_s = gpflow.Param(np.tile(
                    np.eye(self.num_inducing, dtype=settings.float_type),
                    [self.num_outputs, 1, 1]))

        self.mean_function = mean_function

        self._build_cholesky()

    def Kuf(self, ML_Z, NHWC_X):
        """ Returns covariance between inducing points and input.
        Output shape: patch_count x M x N
        """
        # L: patch_length * feature_maps
        NPL_patches = self.extract_patches(NHWC_X)

        N = tf.shape(NHWC_X)[0]
        JL_patches = tf.reshape(NPL_patches, [N * self.patch_count, self.patch_length])
        MJ_Kzx = self.base_kernel.K(ML_Z, JL_patches)
        check_rank = tf.assert_rank(MJ_Kzx, 2)
        with tf.control_dependencies([check_rank]):
            MNP_Kzx = tf.reshape(MJ_Kzx, [self.num_inducing, N, self.patch_count])
            OMN_Kzx = tf.transpose(MNP_Kzx, [2, 0, 1])

        return OMN_Kzx

    def Kff(self, NHWC_X):
        """Kff returns auto covariance of the input.
        :return: O (== P) x N x N covariance matrices.
        """
        NPL_patches = self.extract_patches(NHWC_X)
        PNL_patches = tf.transpose(NPL_patches, [1, 0, 2])
        PNN_Knn = self.base_kernel.K(NPL_patches)
        check_rank = tf.assert_rank(PNN_Knn, 3)
        with tf.control_dependencies([check_rank]):
            return PNN_Knn

    def Kdiag(self, NHWC_X):
        """
        :return: O X N diagonals of the covariance matrices.
        """
        NPL_patches = self.extract_patches(NHWC_X)
        PNL_patches = tf.transpose(NPL_patches, [1, 0, 2])
        def Kdiag(NL_patch):
            ":return: N diagonal of covariance matrix."
            return self.base_kernel.Kdiag(NL_patch)
        return tf.map_fn(Kdiag, PNL_patches)

    def conditional_ND(self, ND_X, full_cov=False):
        """conditional_ND Returns the mean and variance of the normal distribution
        corresponding to q(f) ~ N(Am, Knm + A(S - Kmm)A^T) where A = Knm Kmm^{-1}

        dimensions O: num_outputs

        :param ON_X: The input X of shape O x N
        :param full_cov: if true, var is in (N, N, D_out) instead of (N, D_out) i.e. we
        also compute entries outside the diagonal.
        """
        N = tf.shape(ND_X)[0]
        NHWC_X = tf.reshape(ND_X, [N, self.input_size[0], self.input_size[1], self.feature_maps])
        OMN_Kuf = self.Kuf(self.feature.Z, NHWC_X)
        # NM_Knm = conditionals.Kuf(self.feature, self.conv_kernel, NHWC_X)

        # A = NM_Knm @ MM_Kmm_inv
        OMN_A = tf.matrix_triangular_solve(self.OMM_Lu, OMN_Kuf, lower=True)
        if not self.white:
            # A = A @ Lu^-T
            OMM_Lu = tf.transpose(self.OMM_Lu, [0, 2, 1])
            OMN_A = tf.matrix_triangular_solve(OMM_Lu, OMN_A, lower=False)

        NOM_A = tf.transpose(OMN_A, [2, 0, 1])
        NM1_q_mu = tf.tile(self.M1_q_mu[None, :, :], [N, 1, 1])
        NO1_mean = tf.matmul(NOM_A, NM1_q_mu) # NOM @ NM1 => N x O
        NO_mean = tf.reshape(NO1_mean, [N, self.patch_count])
        # NO_mean = tf.matmul(MN_A, self.MO_q_mu, transpose_a=True) # NM @ MO => NO

        if self.white:
            OMM_SK = -tf.eye(self.num_inducing, dtype=settings.float_type)[None, :, :]
        else:
            OMM_SK = -self.OMM_Ku

        OMM_SK = OMM_SK + tf.matmul(self.OMM_q_s, self.OMM_q_s, transpose_b=True)

        # OMN_A = tf.tile(MN_A[None, :, :], [self.num_outputs, 1, 1])
        OMN_B = OMM_SK @ OMN_A

        if full_cov:
            ONN_Knn = self.Kff(NHWC_X)
            # NN_Knn = self.conv_kernel.K(NHWC_X)
            ONN_var = ONN_Knn + tf.matmul(OMN_A, OMN_B, transpose_a=True)
            var = tf.transpose(ONN_var, [1, 2, 0])
        else:
            ON_Kdiag = self.Kdiag(NHWC_X)
            # ON_Kdiag = self.conv_kernel.Kdiag(NHWC_X)
            OMN_delta_cov = OMN_A * OMN_B
            ON_delta_cov = tf.reduce_sum(OMN_delta_cov, 1)
            ON_var = ON_Kdiag + ON_delta_cov

            var = tf.transpose(ON_var, [1, 0])

        return NO_mean + self.mean_function(ND_X), var

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior

        :return: KL divergence from N(q_mu, q_s) to N(0, I), independently for each GP
        """
        KL = -0.5 * self.num_inducing
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.OMM_q_s) ** 2))

        if not self.white:
            KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.MM_Lu))) * self.num_outputs
            KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.OMM_Lu, self.OMM_q_s, lower=True)))
            M1_Kinv_m = tf.cholesky_solve(self.MM_Lu, self.M1_q_mu)
            KL += 0.5 * tf.reduce_sum(self.M1_q_mu * M1_Kinv_m)
        else:
            KL += 0.5 * tf.reduce_sum(tf.square(self.OMM_q_s))
            KL += 0.5 * tf.reduce_sum(self.M1_q_mu**2)

        return KL

    def _init_q_s_from_Z(self):
        with gpflow.params_as_tensors_for(self.feature):
            MM_Ku = conditionals.Kuu(self.feature, self.conv_kernel, jitter=settings.jitter)
            MM_Lu = tf.linalg.cholesky(MM_Ku)
            MM_Lu = self.enquire_session().run(MM_Lu)
            return gpflow.Param(MM_Lu)

    def _build_cholesky(self):
        with gpflow.params_as_tensors_for(self.feature):
            self.MM_Ku = conditionals.Kuu(self.feature, self.conv_kernel, jitter=settings.jitter)
            self.MM_Lu = tf.linalg.cholesky(self.MM_Ku)
            self.OMM_Lu = tf.tile(self.MM_Lu[None, :, :], [self.num_outputs, 1, 1])
            self.OMM_Ku = tf.tile(self.MM_Ku[None, :, :], [self.num_outputs, 1, 1])

    def _patch_length(self):
        """The number of elements in a patch."""
        return self.feature_maps * np.prod(self.patch_shape)

    def _patch_count(self):
        """The amount of patches in one image."""
        return (self.input_size[0] - self.patch_shape[0] + 1) * (
                self.input_size[1] - self.patch_shape[1] + 1) * self.feature_maps


