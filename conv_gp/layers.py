import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, features, transforms
from gpflow.kullback_leiblers import gauss_kl
from doubly_stochastic_dgp.layers import Layer
from conditionals import base_conditional
from views import FullView

class MultiOutputConvKernel(gpflow.kernels.Kernel):
    def __init__(self, base_kernel, input_dim, patch_count):
        super().__init__(input_dim=input_dim)
        self.base_kernel = base_kernel
        self.patch_count = patch_count

    def Kuu(self, ML_Z):
        M = tf.shape(ML_Z)[0]
        return self.base_kernel.K(ML_Z) + tf.eye(M,
                dtype=settings.float_type) * settings.jitter

    def Kuf(self, features, PNL_patches):
        """ Returns covariance between inducing points and input.
        Output shape: G x P x M x N
        """

        def compute_Kuf(ML_Z):
            def patch_covariance(NL_patches):
                # Returns covariance matrix of size M x N.
                return self.base_kernel.K(ML_Z, NL_patches)
            # out shape: P x M x N
            return tf.map_fn(patch_covariance, PNL_patches, parallel_iterations=self.patch_count)

        return [compute_Kuf(ML_Z) for ML_Z in features.feat_list]

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
    def __init__(self, base_kernel, mean_function, feature=None, view=None,
            white=False,
            gp_count=1,
            **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel

        self.view = view

        self.feature_maps_in = self.view.feature_maps
        self.gp_count = gp_count

        self.patch_count = self.view.patch_count
        self.patch_length = self.view.patch_length
        self.num_outputs = self.patch_count * gp_count

        self.conv_kernel = MultiOutputConvKernel(base_kernel,
                np.prod(view.input_size) * view.feature_maps, patch_count=self.patch_count)

        self.white = white

        self.feature = feature

        self.num_inducing = feature.feat_list[0].shape[0]

        q_mu = np.zeros((self.num_inducing, gp_count), dtype=settings.float_type)
        self.q_mu = gpflow.Param(q_mu)

        #TODO figure out if we need whitened vs non-whitened GP.
        if not self.white:
            GMM_q_sqrt = self._init_q_S()
        else:
            GMM_q_sqrt = np.tile(np.eye(self.num_inducing, dtype=settings.float_type)[None, :, :], [gp_count, 1, 1])
        q_sqrt_transform = gpflow.transforms.LowerTriangular(self.num_inducing, num_matrices=self.gp_count)
        self.q_sqrt = gpflow.Param(GMM_q_sqrt, transform=q_sqrt_transform)

        self.mean_function = mean_function
        self._build_prior_cholesky()

    def conditional_ND(self, ND_X, full_cov=False):
        """
        Returns the mean and the variance of q(f|m, S) = N(f| Am, K_nn + A(S - K_mm)A^T)
        where A = K_nm @ K_mm^{-1}

        dimension O: num_outputs (== patch_count * gp_count)

        :param ON_X: The input X of shape O x N
        :param full_cov: if true, var is in (N, N, D_out) instead of (N, D_out) i.e. we
        also compute entries outside the diagonal.
        """
        N = tf.shape(ND_X)[0]
        NHWC_X = tf.reshape(ND_X, [N, self.view.input_size[0], self.view.input_size[1], self.feature_maps_in])
        PNL_patches = self.view.extract_patches_PNL(NHWC_X)

        GMM_Lu = [tf.cholesky(self.conv_kernel.Kuu(Z)) for Z in self.feature.feat_list]

        GPMN_Kuf = self.conv_kernel.Kuf(self.feature, PNL_patches)

        if full_cov:
            P_Knn = self.conv_kernel.Kff(PNL_patches)
        else:
            P_Knn = self.conv_kernel.Kdiag(PNL_patches)

        def conditional(i):
            MM_Lu = GMM_Lu[i]
            PMN_Kuf = GPMN_Kuf[i]
            q_mu = self.q_mu[:, i][:, None]
            q_sqrt = self.q_sqrt[i, :, :][None, :, :]
            def patch_conditional(tupled):
                MN_Kuf, Knn = tupled
                return base_conditional(MN_Kuf, MM_Lu, Knn, q_mu, full_cov=full_cov,
                        q_sqrt=q_sqrt, white=self.white)
            return tf.map_fn(patch_conditional, (PMN_Kuf, P_Knn), (settings.float_type, settings.float_type),
                    parallel_iterations=self.patch_count)

        means_vars = [conditional(i) for i in range(self.gp_count)]
        mean = [item[0] for item in means_vars]
        var = [item[1] for item in means_vars]

        PNG_mean = tf.concat(mean, axis=2)
        NPG_mean = tf.transpose(PNG_mean, [1, 0, 2])
        mean = tf.reshape(NPG_mean, [N, self.num_outputs])

        if full_cov:
            var = tf.concat(var, axis=1)
            # var: P x G x N x N
            var = tf.transpose(var, [2, 3, 0, 1])
            var = tf.reshape(var, [N, N, self.num_outputs])
        else:
            var = tf.concat(var, axis=2)
            # var: P x N x G
            var = tf.transpose(var, [1, 0, 2])
            var = tf.reshape(var, [N, self.num_outputs])

        mean_view = self.view.mean_view(NHWC_X, PNL_patches)
        mean = mean + self.mean_function(mean_view)
        return mean, var

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior.
        q ~ N(\mu, S)

        :return: KL divergence from q(u) = N(q_mu, q_s) to p(u) ~ N(0, Kuu), independently for each GP
        """
        if self.white:
            return gauss_kl(self.q_mu, self.q_sqrt, K=None)
        else:
            return gauss_kl(self.q_mu, self.q_sqrt, self.GMM_Ku_prior)

    def _build_prior_cholesky(self):
        def compute_Ku(ML_Z):
            return self.conv_kernel.Kuu(ML_Z)
        self.GMM_Ku_prior = tf.stack([
            compute_Ku(Z.value) for Z in self.feature.feat_list
            ], axis=0)

    def _init_q_S(self):
        def compute_Lu(ML_Z):
            MM_Ku = self.conv_kernel.Kuu(ML_Z)
            return tf.linalg.cholesky(MM_Ku)
        GMM_Lu = tf.stack([
            compute_Lu(Z.value) for Z in self.feature.feat_list
            ], axis=0)
        return self.enquire_session().run(GMM_Lu)

