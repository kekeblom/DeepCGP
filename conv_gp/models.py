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

def select_initial_inducing_points(X, M):
    kmeans = cluster.KMeans(n_clusters=M, init='k-means++', n_jobs=-1)
    kmeans.fit(X)
    return kmeans.cluster_centers_

def parse_ints(int_string):
    return [int(i) for i in int_string.split(',')]

def image_HW(patch_count):
    image_height = int(np.sqrt(patch_count))
    return [image_height, image_height]

def identity_conv(NHWC_X, filter_size, feature_maps_in, feature_maps_out, stride):
    conv = IdentityConv2dMean(filter_size, feature_maps_in, feature_maps_out, stride)
    sess = conv.enquire_session()
    return sess.run(conv(NHWC_X))

def build_conv_layer(flags, NHWC_X, M, feature_map, filter_size, stride):
    NHWC = NHWC_X.shape
    view = FullView(input_size=NHWC[1:3],
            filter_size=filter_size,
            feature_maps=NHWC[3],
            stride=stride)

    if flags.identity_mean:
        conv_mean = Conv2dMean(filter_size, NHWC[3], feature_map,
                stride=stride)
    else:
        conv_mean = gpflow.mean_functions.Zero()

    output_shape = image_HW(view.patch_count) + [feature_map]

    H_X = identity_conv(NHWC_X, filter_size, NHWC[3], feature_map, stride)
    conv_features = PatchInducingFeatures.from_images(
            NHWC_X,
            M,
            filter_size)
    conv_mean.set_trainable(False)

    patch_length = filter_size ** 2 * NHWC[3]
    if flags.base_kernel == 'rbf':
        base_kernel = kernels.RBF(patch_length, variance=2.0, lengthscales=2.0)
    elif flags.base_kernel == 'acos':
        base_kernel = kernels.ArcCosine(patch_length, order=0)
    else:
        raise ValueError("Not a valid base-kernel value")

    conv_layer = ConvLayer(
        base_kernel=base_kernel,
        mean_function=conv_mean,
        feature=conv_features,
        view=view,
        white=flags.white,
        gp_count=feature_map)

    # Start with low variance.
    conv_layer.q_sqrt = conv_layer.q_sqrt.value * 1e-5

    return conv_layer, H_X

def build_conv_layers(flags, NHWC_X_train, Ms, feature_maps, strides, filter_sizes):
    H_X = NHWC_X_train
    layers = []
    for (M, feature_map, filter_size, stride) in zip(Ms, feature_maps, filter_sizes, strides):
        conv_layer, H_X = build_conv_layer(flags, H_X, M, feature_map, filter_size, stride)
        layers.append(conv_layer)
    return layers, H_X

def build_last_layer(H_X, M, filter_size, stride, flags):
    NHWC = H_X.shape
    conv_output_count = np.prod(NHWC[1:])
    if flags.last_kernel == 'rbf':
        H_X = H_X.reshape(H_X.shape[0], -1)
        kernel = gpflow.kernels.RBF(conv_output_count, ARD=True)
        Z_rbf = select_initial_inducing_points(H_X, M)
        inducing = features.InducingPoints(Z_rbf)
    else:
        input_dim = filter_size**2 * NHWC[3]
        view = FullView(input_size=NHWC[1:],
                filter_size=filter_size,
                feature_maps=NHWC[3],
                stride=stride)
        inducing = PatchInducingFeatures.from_images(H_X, M, filter_size)
        if flags.last_kernel == 'conv':
            kernel = ConvKernel(
                    base_kernel=gpflow.kernels.RBF(input_dim, variance=2.0, lengthscales=2.0),
                    view=view)
        elif flags.last_kernel == 'add':
            kernel = AdditivePatchKernel(
                    base_kernel=gpflow.kernels.RBF(input_dim, variance=2.0, lengthscales=2.0),
                    view=view)
        else:
            raise ValueError("Invalid last layer kernel")
    return SVGP_Layer(kern=kernel,
                num_outputs=10,
                feature=inducing,
                mean_function=gpflow.mean_functions.Zero(output_dim=10),
                white=flags.white)

def build_model(flags, NHWC_X_train, Y_train):
    Ms = parse_ints(flags.M)
    feature_maps = parse_ints(flags.feature_maps)
    strides = parse_ints(flags.strides)
    filter_sizes = parse_ints(flags.filter_sizes)

    assert len(strides) == len(filter_sizes)

    conv_layers, H_X = build_conv_layers(flags, NHWC_X_train, Ms[0:-1], feature_maps, strides, filter_sizes)

    last_layer = build_last_layer(H_X, Ms[-1], filter_sizes[-1], strides[-1], flags)
    layers = conv_layers + [last_layer]

    X = NHWC_X_train.reshape(-1, np.prod(NHWC_X_train.shape[1:]))
    return DGP_Base(X, Y_train,
            likelihood=gpflow.likelihoods.MultiClass(10),
            num_samples=flags.num_samples,
            layers=layers,
            minibatch_size=flags.batch_size, name='DGP')

