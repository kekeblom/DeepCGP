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

        PMN_Kzx = tf.map_fn(patch_covariance, PNL_patches, parallel_iterations=self.patch_count)
        return PMN_Kzx

    def Kff(self, PNL_patches):
        """Kff returns auto covariance of the input.
        :return: O (== P) x N x N covariance matrices.
        """
        def patch_auto_covariance(NL_patches):
            # Returns covariance matrix of size N x N.
            return self.base_kernel.K(NL_patches)
        return tf.map_fn(patch_auto_covariance, PNL_patches, parallel_iterations=self.patch_count)

    def Kdiag(self, PNL_patches):
        """
        :return: O X N diagonals of the covariance matrices.
        """
        def Kdiag(NL_patch):
            ":return: N diagonal of covariance matrix."
            return self.base_kernel.Kdiag(NL_patch)
        return tf.map_fn(Kdiag, PNL_patches, parallel_iterations=self.patch_count)

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
        Returns the mean and the variance of q(f|m, S) = N(f| Am, K_nn + A(S - K_mm)A^T)
        where A = K_nm @ K_mm^{-1}

        dimension O: num_outputs (== patch_count)

        :param ON_X: The input X of shape O x N
        :param full_cov: if true, var is in (N, N, D_out) instead of (N, D_out) i.e. we
        also compute entries outside the diagonal.
        """

        N = tf.shape(ND_X)[0]
        P = self.patch_count
        NHWC_X = tf.reshape(ND_X, [N, self.input_size[0], self.input_size[1], self.feature_maps])
        PNL_patches = self.conv_kernel.extract_patches_PNL(NHWC_X)

        MM_Kuu = self.conv_kernel.Kuu(self.feature.Z)
        MM_Lu = tf.linalg.cholesky(MM_Kuu)
        IMM_q_S = tf.matmul(self.IMM_q_sqrt, self.IMM_q_sqrt, transpose_b=True)

        def solve_A(MN_Kuf):
            MN_A = tf.matrix_triangular_solve(MM_Lu, MN_Kuf, lower=True)
            if not self.white:
                MM_Lu_t = tf.transpose(MM_Lu, [1, 0])
                MN_A = tf.matrix_triangular_solve(MM_Lu_t, MN_A, lower=False)
            return MN_A

        PMN_Kuf = self.conv_kernel.Kuf(self.feature.Z, PNL_patches)
        PMN_A = tf.map_fn(solve_A, PMN_Kuf, parallel_iterations=self.patch_count)

        PNM_A = tf.transpose(PMN_A, [0, 2, 1])

        def compute_mean(NM_A):
            return tf.matmul(NM_A, self.M1_q_mu)[:, 0]

        PN_mean = tf.map_fn(compute_mean, PNM_A, parallel_iterations=self.patch_count)
        NP_mean = tf.transpose(PN_mean, [1, 0])

        MM_B = IMM_q_S[0, :, :] - MM_Kuu

        def compute_additive(NM_A):
            return tf.matmul(tf.matmul(NM_A, MM_B), NM_A, transpose_b=True)

        PNN_additive = tf.map_fn(compute_additive, PNM_A, parallel_iterations=self.patch_count)

        if full_cov:
            PNN_Knn = self.conv_kernel.Kff(PNL_patches)
            PNN_var = PNN_Knn + PNN_additive
            var = tf.transpose(PNN_var, [1, 2, 0])
        else:
            PN_Kdiag = self.conv_kernel.Kdiag(PNL_patches)
            PN_diag = tf.matrix_diag_part(PNN_additive)

            PN_var = PN_Kdiag + PN_diag
            var = tf.transpose(PN_var, [1, 0])

        return NP_mean + self.mean_function(NHWC_X), var

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
            return 0.5 * KL
        else:
            return gauss_kl(self.M1_q_mu, self.IMM_q_sqrt, self.MM_Ku_prior)


    def _build_prior_cholesky(self):
        self.MM_Ku_prior = self.conv_kernel.Kuu(self.feature.Z.parameter_tensor)
        MM_Lu_prior = tf.linalg.cholesky(self.MM_Ku_prior)
        self.MM_Lu_prior = self.enquire_session().run(MM_Lu_prior)

    def _init_q_S(self):
        MM_Ku = self.conv_kernel.Kuu(self.feature.Z.parameter_tensor)
        MM_Lu = tf.linalg.cholesky(MM_Ku)
        MM_Lu = self.enquire_session().run(MM_Lu)
        return MM_Lu


