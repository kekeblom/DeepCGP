import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, features, conditionals, transforms
from gpflow.kullback_leiblers import gauss_kl
from doubly_stochastic_dgp.layers import Layer
from kernels import PatchMixin, PatchInducingFeature

class MultiOutputConvKernel(gpflow.kernels.Kernel, PatchMixin):
    def __init__(self, base_kernel, input_size, filter_size, feature_maps):
        super().__init__(input_dim=np.prod(input_size))
        self.base_kernel = base_kernel
        self.input_size = input_size
        self.filter_size = filter_size
        self.stride = 1
        self.dilation = 1
        self.feature_maps = feature_maps
        self.patch_shape = (filter_size, filter_size)
        self.patch_count = self._patch_count()
        self.patch_length = self._patch_length()

    def Kuu(self, ML_Z):
        M = tf.shape(ML_Z)[0]
        return self.base_kernel.K(ML_Z) + tf.eye(M,
                dtype=settings.float_type) * settings.jitter

    def Kuf(self, ML_Z, PNL_patches):
        """ Returns covariance between inducing points and input.
        Output shape: patch_count x M x N
        """
        def patch_covariance(NL_patches):
            # Returns covariance matrix of size M x N.
            return self.base_kernel.K(ML_Z, NL_patches)

        PMN_Kzx = tf.map_fn(patch_covariance, PNL_patches)
        return PMN_Kzx

    def Kff(self, PNL_patches):
        """Kff returns auto covariance of the input.
        :return: O (== P) x N x N covariance matrices.
        """
        def patch_auto_covariance(NL_patches):
            # Returns covariance matrix of size N x N.
            return self.base_kernel.K(NL_patches)
        return tf.map_fn(patch_auto_covariance, PNL_patches)

    def Kdiag(self, PNL_patches):
        """
        :return: O X N diagonals of the covariance matrices.
        """
        def Kdiag(NL_patch):
            ":return: N diagonal of covariance matrix."
            return self.base_kernel.Kdiag(NL_patch)
        return tf.map_fn(Kdiag, PNL_patches)

class ConvLayer(Layer):
    def __init__(self, base_kernel, mean_function, feature=None,
            input_size=None,
            feature_maps=None,
            filter_size=None,
            white=False,
            **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.input_size = input_size
        self.feature_maps = feature_maps

        self.conv_kernel = MultiOutputConvKernel(base_kernel, input_size, filter_size, feature_maps)

        self.patch_count = self.conv_kernel._patch_count()
        self.patch_length = self.conv_kernel._patch_length()

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
        self._build_prior_cholesky()

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
        PNL_patches = self.conv_kernel.extract_patches_PNL(NHWC_X)

        PMN_Kuf = self.conv_kernel.Kuf(self.feature.Z, PNL_patches)

        MM_Kuu = self.conv_kernel.Kuu(self.feature.Z)
        MM_Lu = tf.linalg.cholesky(MM_Kuu)
        PMM_Lu = tf.tile(MM_Lu[None, :, :], [self.patch_count, 1, 1])

        PMN_A = tf.matrix_triangular_solve(PMM_Lu, PMN_Kuf, lower=True)
        if not self.white:
            PMM_Lu_t = tf.transpose(PMM_Lu, [0, 2, 1])
            PMN_A = tf.matrix_triangular_solve(PMM_Lu_t, PMN_A, lower=False)

        PNM_A = tf.transpose(PMN_A, [0, 2, 1])

        PM1_q_mu = tf.tile(self.M1_q_mu[None, :, :], [self.patch_count, 1, 1])
        PN_mean = tf.matmul(PNM_A, PM1_q_mu)[:, :, 0]
        NP_mean = tf.transpose(PN_mean, [1, 0])

        IMM_q_S = tf.matmul(self.IMM_q_sqrt, self.IMM_q_sqrt, transpose_b=True)

        PMM_SK = tf.tile(IMM_q_S - MM_Kuu, [self.patch_count, 1, 1])

        PNN_additive = PNM_A @ PMM_SK @ PMN_A

        if full_cov:
            PNN_Knn = self.conv_kernel.Kff(PNL_patches)
            PNN_var = PNN_Knn + PNN_additive
            var = tf.transpose(PNN_var, [1, 2, 0])
        else:
            PN_Kdiag = self.conv_kernel.Kdiag(PNL_patches)
            PN_diag = tf.matrix_diag_part(PNN_additive)

            PN_var = PN_Kdiag + PN_diag
            var = tf.transpose(PN_var, [1, 0])

        return NP_mean + self.mean_function(ND_X), var

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior.
        q ~ N(\mu, S)
        if white:
            KL(q||N(0, I)) = 0.5 * (tr(S) + \mu^T\mu - k - \sum diag(S))
        else:
            TODO

        :return: KL divergence from q(u) = N(q_mu, q_s) to p(u) ~ N(0, Kuu), independently for each GP
        """
        k = self.num_inducing # Dimensionality of the distributions.
        if self.white:
            KL = -k
            KL += tf.reduce_sum(tf.matrix_diag_part(self.IMM_q_sqrt))
            KL += tf.reduce_sum(tf.square(self.M1_q_mu))
            KL -= tf.reduce_sum(tf.matrix_diag_part(self.MM_Lu))
            return 0.5 * KL * self.patch_count # Once kfor each output.
        else:
            # KL = -k

            # MM_q_sqrt = self.IMM_q_sqrt[0, :, :]
            # MM_q_S = tf.matmul(MM_q_sqrt, MM_q_sqrt, transpose_b=True)
            # # Trace term
            # MM_trace_inner = tf.cholesky_solve(self.MM_Lu, MM_q_S)
            # KL += tf.trace(MM_trace_inner)
            # # q_mu^T @ Kuu @ q_mu
            # KL += tf.reduce_sum(tf.matmul(self.M1_q_mu, self.MM_Ku_prior, transpose_a=True) @ self.M1_q_mu, [0, 1])
            # # log determinant term
            # KL += 2 * tf.reduce_sum(tf.log(tf.diag_part(self.MM_Lu)))
            # KL -= 2 * tf.reduce_sum(tf.log(tf.diag_part(MM_q_sqrt)))

            # return 0.5 * KL * self.patch_count
            return gauss_kl(self.M1_q_mu, self.IMM_q_sqrt, self.MM_Ku_prior) * self.patch_count

    def _build_prior_cholesky(self):
        self.MM_Ku_prior = self.conv_kernel.Kuu(self.feature.Z.parameter_tensor)
        MM_Lu_prior = tf.linalg.cholesky(self.MM_Ku_prior)
        self.MM_Lu_prior = self.enquire_session().run(MM_Lu_prior)

    def _init_q_S(self):
        MM_Ku = self.conv_kernel.Kuu(self.feature.Z.parameter_tensor)
        MM_Lu = tf.linalg.cholesky(MM_Ku)
        MM_Lu = self.enquire_session().run(MM_Lu)
        return MM_Lu


