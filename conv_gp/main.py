import math
import numpy as np
import tensorflow as tf
import gpflow
import observations
import utils
from sklearn import preprocessing, decomposition, cluster
from gpflow import settings, kernels, features
from doubly_stochastic_dgp.dgp import DGP_Base
from doubly_stochastic_dgp.layers import SVGP_Layer
from kernels import ConvKernel, IndependentPatchInducingFeatures, PatchInducingFeatures
from layers import ConvLayer
from views import FullView, RandomPartialView
from mean_functions import Conv2dMean, IdentityConv2dMean
from gpflow.actions import Loop

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

    conv_mean = gpflow.mean_functions.Zero()

    output_shape = image_HW(view.patch_count) + [feature_map]

    H_X = identity_conv(NHWC_X, filter_size, NHWC[3], feature_map, stride)
    conv_features = IndependentPatchInducingFeatures.from_images(
            NHWC_X,
            M,
            filter_size,
            count=feature_map)
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
        white=False,
        gp_count=feature_map)

    # Start with low variance.
    conv_layer.q_sqrt = conv_layer.q_sqrt.value * 1e-5

    return conv_layer, H_X

def build_conv_layers(flags, NHWC_X_train, Ms):
    feature_maps = parse_ints(flags.feature_maps)
    filter_sizes = parse_ints(flags.filter_sizes)
    strides = parse_ints(flags.strides)
    H_X = NHWC_X_train
    layers = []
    for (M, feature_map, filter_size, stride) in zip(Ms, feature_maps, filter_sizes, strides):
        conv_layer, H_X = build_conv_layer(flags, H_X, M, feature_map, filter_size, stride)
        layers.append(conv_layer)
    return layers, H_X

def build_last_layer(H_X, M, flags):
    NHWC = H_X.shape
    conv_output_count = np.prod(NHWC[1:])
    if flags.last_kernel == 'rbf':
        H_X = H_X.reshape(H_X.shape[0], -1)
        kernel = gpflow.kernels.RBF(conv_output_count, ARD=True)
        Z_rbf = select_initial_inducing_points(H_X, M)
        inducing = features.InducingPoints(Z_rbf)
    elif flags.last_kernel == 'conv':
        filter_size = 5
        input_dim = filter_size**2 * NHWC[3]
        view = FullView(input_size=NHWC[1:],
                filter_size=filter_size,
                feature_maps=NHWC[3],
                stride=1)
        kernel = ConvKernel(
                base_kernel=gpflow.kernels.RBF(input_dim, variance=2.0, lengthscales=2.0),
                view=view)
        inducing = PatchInducingFeatures.from_images(H_X, M, filter_size)
    else:
        raise ValueError("Invalid last layer kernel")
    return SVGP_Layer(kern=kernel,
                num_outputs=10,
                feature=inducing,
                mean_function=gpflow.mean_functions.Zero(output_dim=10),
                white=False)

def build_model(flags, X_train, Y_train):
    NHWC_X_train = X_train.reshape(-1, 28, 28, 1)
    Ms = parse_ints(flags.M)

    conv_layers, H_X = build_conv_layers(flags, NHWC_X_train, Ms[0:-1])

    last_layer = build_last_layer(H_X, Ms[-1], flags)
    layers = conv_layers + [last_layer]

    return DGP_Base(X_train, Y_train,
            likelihood=gpflow.likelihoods.MultiClass(10),
            num_samples=flags.num_samples,
            layers=layers,
            minibatch_size=flags.batch_size)

class MNIST(object):
    def __init__(self, flags):
        self.flags = flags

        self._load_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logger()

    def conclude(self):
        self.log.close()

    def train_step(self):
        self._optimize()
        self._log_step()
        self._checkpoint_model()

    def _log_step(self):
        entry = self.log.write_entry(self.model)
        self.tensorboard_log.write_entry(self.model)
        print(entry)

    def _optimize(self):
        numiter = self.flags.test_every
        Loop(self.optimizers, stop=numiter)()

    def _checkpoint_model(self):
        saver = tf.train.Saver()
        model_session = self.model.enquire_session()
        saver.save(model_session, self.flags.model_path)

    def _setup_model(self):
        self.model = build_model(self.flags, self.X_train, self.Y_train)
        self.model.compile()

    def _setup_learning_rate(self):
        self.learning_rate = tf.train.exponential_decay(self.flags.lr, global_step=self.global_step,
                decay_rate=0.1, decay_steps=self.flags.lr_decay_steps)

    def _setup_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        self._setup_learning_rate()
        self.model.enquire_session().run(self.global_step.initializer)

        self.optimizers = []
        if self.flags.optimizer == "NatGrad":
            variational_parameters = [(self.model.layers[-1].q_mu, self.model.layers[-1].q_sqrt)]

            for params in variational_parameters:
                for param in params:
                    param.set_trainable(False)

            nat_grad = gpflow.train.NatGradOptimizer(gamma=self.flags.gamma).make_optimize_action(self.model,
                    var_list=variational_parameters)
            self.optimizers.append(nat_grad)

        if self.flags.optimizer == "SGD":
            opt = gpflow.train.GradientDescentOptimizer(learning_rate=self.learning_rate)\
                    .make_optimize_action(self.model, global_step=self.global_step)
            self.optimizers.append(opt)
        elif self.flags.optimizer == "Adam" or self.flags.optimizer == "NatGrad":
            opt = gpflow.train.AdamOptimizer(learning_rate=self.learning_rate).make_optimize_action(self.model,
                    global_step=self.global_step)
            self.optimizers.append(opt)

        if self.flags.optimizer not in ["Adam", "NatGrad", "SGD"]:
            raise ValueError("Not a supported optimizer. Try Adam or NatGrad.")

    def _load_data(self):
        if self.flags.fashion:
            load_fn = observations.fashion_mnist
        else:
            load_fn = observations.mnist
        (self.X_train, self.Y_train), (
                self.X_test, self.Y_test) = load_fn('/tmp/mnist/')
        self.Y_train = self.Y_train.reshape(-1, 1)
        self.Y_test = self.Y_test.reshape(-1, 1)
        self._select_training_points()
        self._select_test_points()
        self._preprocess_data()

    def _select_training_points(self):
        chosen = slice(0, self.flags.N)
        self.X_train = self.X_train[chosen, :]
        self.Y_train = self.Y_train[chosen, :]

    def _select_test_points(self):
        arange = np.arange(0, len(self.X_test))
        chosen = np.random.choice(arange, self.flags.test_size, replace=False)
        self.X_test = self.X_test[chosen, :]
        self.Y_test = self.Y_test[chosen, :]

    def _preprocess_data(self):
        self.X_transform = preprocessing.StandardScaler()
        self.X_train = self.X_transform.fit_transform(self.X_train).astype(settings.float_type)
        self.X_test = self.X_transform.transform(self.X_test).astype(settings.float_type)

    def _setup_logger(self):
        self._init_logger()
        self._init_tensorboard()

    def _init_logger(self):
        loggers = [
            utils.GlobalStepLogger(),
            utils.AccuracyLogger(self.X_test, self.Y_test),
        ]
        self.log = utils.Log(self.flags.log_dir,
                self.flags.name,
                loggers)
        self.log.write_flags(self.flags)

    def _init_tensorboard(self):
        sample_task = utils.LayerOutputLogger(self.model, self.X_test)
        model_parameter_task = utils.ModelParameterLogger(self.model)
        likelihood = utils.LogLikelihoodLogger()

        tasks = [sample_task, likelihood, model_parameter_task]
        self.tensorboard_log = utils.TensorBoardLog(tasks, self.flags.tensorboard_dir, self.flags.name,
                self.model, self.global_step)

    def _write_initial_inducing_points(self):
        self.log.write_inducing_points(self.model, "z_init.npy")


def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fashion', action='store_true', default=False,
            help="Use fashion MNIST instead of regular MNIST.")
    parser.add_argument('-N', type=int,
            help="How many training examples to use.", default=60000)
    parser.add_argument('--name', type=str, required=True,
            help="What to call the experiment. Determines the directory where results are dumped.")
    parser.add_argument('--lr-decay-steps', type=int, default=50000,
            help="The program uses exponential learning rate decay with 0.1 decay every lr-decay-steps.")
    parser.add_argument('--test-every', type=int, default=5000,
            help="How often to evaluate the test accuracy. Unit optimization iterations.")
    parser.add_argument('--test-size', type=int, default=10000)
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--log-dir', type=str, default='results',
            help="Directory to write the results to.")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128,
            help="Minibatch size to use in optimization.")
    parser.add_argument('--tensorboard-dir', type=str, default='/tmp/mnist/tensorboard')
    parser.add_argument('--optimizer', type=str, default='Adam',
            help="Either Adam or NatGrad")
    parser.add_argument('--model-path', default='/tmp/mnist/model.ckpt')

    parser.add_argument('-M', type=str, default='64',
            help="How many inducing points to use at each layer.")
    parser.add_argument('--feature-maps', type=str, default='1')
    parser.add_argument('--filter-sizes', type=str, default='5')
    parser.add_argument('--strides', type=str, default='1')
    parser.add_argument('--base-kernel', type=str, default='rbf')

    parser.add_argument('--last-kernel', type=str, default='rbf')

    parser.add_argument('--gamma', type=float, default=1.0,
            help="Gamma parameter to start with for natgrad.")

    return parser.parse_args()

def train_steps(flags):
    # Roughly until the learning rate becomes 1e-5
    decay_count = math.log(1e-5 / flags.lr, 0.1) # How many times decay has to be applied to reach 1e-5.
    return math.ceil(flags.lr_decay_steps * decay_count / flags.test_every)

def main():
    flags = read_args()

    experiment = MNIST(flags)

    try:
        for i in range(train_steps(flags)):
            experiment.train_step()
    finally:
        experiment.conclude()

if __name__ == "__main__":
    main()


