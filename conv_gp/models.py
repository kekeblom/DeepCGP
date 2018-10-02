import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, kernels, features

from doubly_stochastic_dgp.dgp import DGP_Base
from doubly_stochastic_dgp.layers import SVGP_Layer
from kernels import ConvKernel, PatchInducingFeatures, AdditivePatchKernel
from layers import ConvLayer
from views import FullView, RandomPartialView
from mean_functions import Conv2dMean, IdentityConv2dMean
from sklearn import cluster

def parse_ints(int_string):
    if int_string == '':
        return []
    else:
        return [int(i) for i in int_string.split(',')]

def image_HW(patch_count):
    image_height = int(np.sqrt(patch_count))
    return [image_height, image_height]

def select_initial_inducing_points(X, M):
    kmeans = cluster.KMeans(n_clusters=M, init='k-means++', n_jobs=-1)
    kmeans.fit(X)
    return kmeans.cluster_centers_

def identity_conv(NHWC_X, filter_size, feature_maps_in, feature_maps_out, stride):
    conv = IdentityConv2dMean(filter_size, feature_maps_in, feature_maps_out, stride)
    sess = conv.enquire_session()
    random_images = np.random.choice(np.arange(NHWC_X.shape[0]), size=1000)
    return sess.run(conv(NHWC_X[random_images]))

class ModelBuilder(object):
    def __init__(self, flags, NHWC_X_train, Y_train, model_path=None):
        self.flags = flags
        self.X_train = NHWC_X_train
        self.Y_train = Y_train
        self.model_path = model_path
        self.global_step = None

    def build(self):
        Ms = parse_ints(self.flags.M)
        feature_maps = parse_ints(self.flags.feature_maps)
        strides = parse_ints(self.flags.strides)
        filter_sizes = parse_ints(self.flags.filter_sizes)

        loaded_parameters = {}
        if self.flags.load_model is not None:
            global_step, loaded_parameters = self._load_layer_parameters(Ms)
            self.global_step = global_step

        assert len(strides) == len(filter_sizes)
        assert len(feature_maps) == (len(Ms) - 1)

        conv_layers, H_X = self._conv_layers(Ms[0:-1], feature_maps, strides, filter_sizes,
                loaded_parameters)

        last_layer_parameters = self._last_layer_parameters(loaded_parameters)
        last_layer = self._last_layer(H_X, Ms[-1], filter_sizes[-1], strides[-1],
                last_layer_parameters)
        layers = conv_layers + [last_layer]

        X = self.X_train.reshape(-1, np.prod(self.X_train.shape[1:]))
        return DGP_Base(X, self.Y_train,
                likelihood=gpflow.likelihoods.MultiClass(10),
                num_samples=self.flags.num_samples,
                layers=layers,
                minibatch_size=self.flags.batch_size, name='DGP')

    def _conv_layers(self, Ms, feature_maps, strides, filter_sizes, loaded_parameters={}):
        H_X = self.X_train
        layers = []
        for i in range(len(feature_maps)):
            M = Ms[i]
            feature_map = feature_maps[i]
            filter_size = filter_sizes[i]
            stride = strides[i]
            layer_params = loaded_parameters.get(i)

            conv_layer, H_X = self._conv_layer(H_X, M, feature_map, filter_size, stride, layer_params)
            layers.append(conv_layer)
        return layers, H_X

    def _conv_layer(self, NHWC_X, M, feature_map, filter_size, stride, layer_params=None):
        if layer_params is None:
            layer_params = {}
        NHWC = NHWC_X.shape
        view = FullView(input_size=NHWC[1:3],
                filter_size=filter_size,
                feature_maps=NHWC[3],
                stride=stride)

        if self.flags.identity_mean:
            conv_mean = Conv2dMean(filter_size, NHWC[3], feature_map,
                    stride=stride)
        else:
            conv_mean = gpflow.mean_functions.Zero()
        conv_mean.set_trainable(False)

        output_shape = image_HW(view.patch_count) + [feature_map]

        H_X = identity_conv(NHWC_X, filter_size, NHWC[3], feature_map, stride)
        if len(layer_params) == 0:
            conv_features = PatchInducingFeatures.from_images(
                    NHWC_X,
                    M,
                    filter_size)
        else:
            conv_features = PatchInducingFeatures(layer_params.get('Z'))

        patch_length = filter_size ** 2 * NHWC[3]
        if self.flags.base_kernel == 'rbf':
            lengthscales = layer_params.get('base_kernel/lengthscales', 5.0)
            variance = layer_params.get('base_kernel/variance', 5.0)
            base_kernel = kernels.RBF(patch_length, variance=variance, lengthscales=lengthscales)
        elif self.flags.base_kernel == 'acos':
            base_kernel = kernels.ArcCosine(patch_length, order=0)
        else:
            raise ValueError("Not a valid base-kernel value")

        q_mu = layer_params.get('q_mu')
        q_sqrt = layer_params.get('q_sqrt')

        conv_layer = ConvLayer(
            base_kernel=base_kernel,
            mean_function=conv_mean,
            feature=conv_features,
            view=view,
            white=self.flags.white,
            gp_count=feature_map,
            q_mu=q_mu,
            q_sqrt=q_sqrt)

        if q_sqrt is None:
            # Start with low variance.
            conv_layer.q_sqrt = conv_layer.q_sqrt.value * 1e-5

        return conv_layer, H_X

    def _last_layer(self, H_X, M, filter_size, stride, layer_params=None):
        if layer_params is None:
            layer_params = {}

        NHWC = H_X.shape
        conv_output_count = np.prod(NHWC[1:])
        Z = layer_params.get('Z')
        q_mu = layer_params.get('q_mu')
        q_sqrt = layer_params.get('q_sqrt')

        if Z is not None:
            saved_filter_size = int(np.sqrt(Z.shape[1] / NHWC[3]))
            if filter_size != saved_filter_size:
                print("filter_size {} != {} for last layer. Resetting parameters.".format(filter_size, saved_filter_size))
                Z = None
                q_mu = None
                q_sqrt = None

        if self.flags.last_kernel == 'rbf':
            H_X = H_X.reshape(H_X.shape[0], -1)
            lengthscales = layer_params.get('lengthscales', 5.0)
            variance = layer_params.get('variance', 5.0)
            kernel = gpflow.kernels.RBF(conv_output_count, lengthscales=lengthscales, variance=variance,
                    ARD=True)
            if Z is None:
                Z = select_initial_inducing_points(H_X, M)
            inducing = features.InducingPoints(Z)
        else:
            lengthscales = layer_params.get('base_kernel/lengthscales', 5.0)
            variance = layer_params.get('base_kernel/variance', 5.0)
            input_dim = filter_size**2 * NHWC[3]
            view = FullView(input_size=NHWC[1:],
                    filter_size=filter_size,
                    feature_maps=NHWC[3],
                    stride=stride)
            if Z is None:
                inducing = PatchInducingFeatures.from_images(H_X, M, filter_size)
            else:
                inducing = PatchInducingFeatures(Z)
            patch_weights = layer_params.get('patch_weights')
            if self.flags.last_kernel == 'conv':
                kernel = ConvKernel(
                        base_kernel=gpflow.kernels.RBF(input_dim, variance=variance, lengthscales=lengthscales),
                        view=view, patch_weights=patch_weights)
            elif self.flags.last_kernel == 'add':
                kernel = AdditivePatchKernel(
                        base_kernel=gpflow.kernels.RBF(input_dim, variance=variance, lengthscales=lengthscales),
                        view=view, patch_weights=patch_weights)
            else:
                raise ValueError("Invalid last layer kernel")
        return SVGP_Layer(kern=kernel,
                    num_outputs=10,
                    feature=inducing,
                    mean_function=gpflow.mean_functions.Zero(output_dim=10),
                    white=self.flags.white,
                    q_mu=q_mu,
                    q_sqrt=q_sqrt)

    def _load_layer_parameters(self, Ms):
        parameters = np.load(self.model_path).item()
        global_step = parameters['global_step']
        del parameters['global_step']
        layer_params = {}

        def parse_layer_path(key):
            if 'layers' not in key:
                return None, None
            parts = key.split('/')
            return int(parts[2]), "/".join(parts[3:])

        for key, value in parameters.items():
            layer, path = parse_layer_path(key)
            if layer is None:
                continue
            layer_values = layer_params.get(layer, {})
            if 'q_mu' in path:
                layer_values['q_mu'] = value
            elif 'q_sqrt' in path:
                layer_values['q_sqrt'] = value
            elif 'Z' in path:
                layer_values['Z'] = value
            elif 'base_kernel/variance' in path:
                layer_values['base_kernel/variance'] = value
            elif 'base_kernel/lengthscales' in path:
                layer_values['base_kernel/lengthscales'] = value
            elif 'patch_weights' in path:
                layer_values['patch_weights'] = value
            layer_params[layer] = layer_values

        stored_layers = max(layer_params.keys()) + 1
        model_layers = len(Ms)
        assert stored_layers <= model_layers, "Can't load model if "
        if stored_layers != model_layers:
            last_layer = stored_layers - 1
            last_layer_params = layer_params[last_layer]
            del layer_params[last_layer]
            layer_params[model_layers-1] = last_layer_params

        return global_step, layer_params

    def _last_layer_parameters(self, layer_params):
        keys = list(layer_params.keys())
        if len(keys) > 0:
            return layer_params[max(keys)]
        else:
            return None

