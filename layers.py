import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, features, conditionals, transforms
from gpflow.dispatch import dispatch
from gpflow.kullback_leiblers import gauss_kl
from doubly_stochastic_dgp.layers import Layer
from kernels import PatchMixin, PatchInducingFeature

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

        self.feature = feature

        self.num_inducing = len(feature)
        self.num_outputs = self.patch_count

        M1_q_mu = np.zeros((self.num_inducing, 1), dtype=settings.float_type)
        self.M1_q_mu = gpflow.Param(M1_q_mu)

        #TODO figure out if we need whitened vs non-whitened GP.
        if not self.white:
            MM_q_sqrt = self._init_q_S()
        else:
            MM_q_sqrt = np.eye(self.num_inducing, dtype=settings.float_type)
        q_sqrt_transform = gpflow.transforms.LowerTriangular(self.num_inducing)
        self.IMM_q_sqrt = gpflow.Param(MM_q_sqrt[None, :, :], transform=q_sqrt_transform)

        self.mean_function = mean_function

        self._build_cholesky()

    def Kuu(self):
        return self.base_kernel.K(self.feature.Z) + tf.eye(self.num_inducing,
                dtype=settings.float_type) * settings.jitter

    def Kuf(self, ML_Z, NHWC_X):
        """ Returns covariance between inducing points and input.
        Output shape: patch_count x M x N
        """
        # L: patch_length * feature_maps
        NPL_patches = self.extract_patches(NHWC_X)

        N = tf.shape(NHWC_X)[0]
        JL_patches = tf.reshape(NPL_patches, [N * self.patch_count, self.patch_length])
        MJ_Kzx = self.base_kernel.K(ML_Z, JL_patches)

        check_shape = tf.assert_equal(tf.shape(MJ_Kzx)[0], self.num_inducing)
        check_shape2 = tf.assert_equal(tf.shape(MJ_Kzx)[1], self.patch_count * N)
        check_rank = tf.assert_rank(MJ_Kzx, 2)

        with tf.control_dependencies([check_rank, check_shape, check_shape2]):
            MNP_Kzx = tf.reshape(MJ_Kzx, [self.num_inducing, N, self.patch_count])
            PMN_Kzx = tf.transpose(MNP_Kzx, [2, 0, 1])

            return PMN_Kzx

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
        """
        Returns the mean and the variance of q(f|m, S) = N(f| Am, K_nn + A(S - K_mn)A^T)
        where A = K_nm @ K_mm^{-1}

        dimension O: num_outputs (== patch_count)

        :param ON_X: The input X of shape O x N
        :param full_cov: if true, var is in (N, N, D_out) instead of (N, D_out) i.e. we
        also compute entries outside the diagonal.
        """

        N = tf.shape(ND_X)[0]
        NHWC_X = tf.reshape(ND_X, [N, self.input_size[0], self.input_size[1], self.feature_maps])
        OMN_Kuf = self.Kuf(self.feature.Z, NHWC_X)

        OMN_A = tf.matrix_triangular_solve(self.OMM_Lu, OMN_Kuf, lower=True)
        if not self.white:
            OMM_Lu = tf.transpose(self.OMM_Lu, [0, 2, 1])
            OMN_A = tf.matrix_triangular_solve(OMM_Lu, OMN_A, lower=False)

        ONM_A = tf.transpose(OMN_A, [0, 2, 1])

        OM1_q_mu = tf.tile(self.M1_q_mu[None, :, :], [self.patch_count, 1, 1])
        ON_mean = tf.matmul(ONM_A, OM1_q_mu)[:, :, 0]
        NO_mean = tf.transpose(ON_mean, [1, 0])

        IMM_q_S = tf.matmul(self.IMM_q_sqrt, self.IMM_q_sqrt, transpose_b=True)

        OMM_SK = tf.tile(IMM_q_S - self.MM_Ku, [self.patch_count, 1, 1])

        ONN_additive = ONM_A @ OMM_SK @ OMN_A

        if full_cov:
            ONN_Knn = self.Kff(NHWC_X)
            ONN_var = ONN_Knn + ONN_additive
            var = tf.transpose(ONN_var, [1, 2, 0])
        else:
            ON_Kdiag = self.Kdiag(NHWC_X)
            ON_diag = tf.matrix_diag_part(ONN_additive)

            ON_var = ON_Kdiag + ON_diag
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

    def _init_q_S(self):
        with gpflow.params_as_tensors_for(self.feature):
            MM_Ku = self.Kuu()
            MM_Lu = tf.linalg.cholesky(MM_Ku)
            MM_Lu = self.enquire_session().run(MM_Lu)
            return MM_Lu

    def _build_cholesky(self):
        with gpflow.params_as_tensors_for(self.feature):
            self.MM_Ku = self.Kuu()
            self.MM_Lu = tf.linalg.cholesky(self.MM_Ku)
            self.OMM_Lu = tf.tile(self.MM_Lu[None, :, :], [self.patch_count, 1, 1])
            self.OMM_Ku = tf.tile(self.MM_Ku[None, :, :], [self.patch_count, 1, 1])

    def _patch_length(self):
        """The number of elements in a patch."""
        return self.feature_maps * np.prod(self.patch_shape)

    def _patch_count(self):
        """The amount of patches in one image."""
        return (self.input_size[0] - self.patch_shape[0] + 1) * (
                self.input_size[1] - self.patch_shape[1] + 1) * self.feature_maps


