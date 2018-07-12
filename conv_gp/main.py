import math
import numpy as np
import tensorflow as tf
import gpflow
import observations
import utils
from sklearn import preprocessing, decomposition, cluster
from gpflow import settings, features, kernels
from gpflow.decors import params_as_tensors
from doubly_stochastic_dgp.dgp import DGP_Base
from doubly_stochastic_dgp.layers import SVGP_Layer
from kernels import ConvKernel, PatchInducingFeature
from layers import ConvLayer

def select_initial_inducing_points(X, M):
    kmeans = cluster.KMeans(n_clusters=M, init='k-means++', n_jobs=-1)
    kmeans.fit(X)
    return kmeans.cluster_centers_

class Conv2dMean(gpflow.mean_functions.MeanFunction):
    def __init__(self, filter_size, feature_maps):
        super().__init__()
        self.filter_size = filter_size
        self.feature_maps = feature_maps
        self.conv_filter = gpflow.Param(self._init_filter())

    @params_as_tensors
    def __call__(self, NHWC_X):
        convolved = tf.nn.conv2d(NHWC_X, self.conv_filter, [1, 1, 1, 1], "VALID",
                data_format="NHWC")
        NHWC = tf.shape(NHWC_X)
        return tf.reshape(convolved, [NHWC[0], self._output_length(NHWC)])

    def _output_length(self, NHWC):
        return (NHWC[1] - self.filter_size + 1) * (
                NHWC[1] - self.filter_size + 1) * self.feature_maps

    def _init_filter(self):
        # Only supports square filters with odd size for now.
        identity_filter = np.zeros((self.filter_size, self.filter_size), dtype=gpflow.settings.float_type)
        identity_filter[self.filter_size // 2, self.filter_size // 2] = 1.0
        return np.tile(identity_filter[:, :, None, None], [1, 1, self.feature_maps, self.feature_maps])


class MNIST(object):
    def __init__(self, flags):
        self.flags = flags

        self._load_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logger()
        # self._write_initial_inducing_points()

    def conclude(self):
        # self.log.write_model(self.model)
        self.log.close()

    def train_step(self):
        self._optimize()
        self._log_step()

    def _log_step(self):
        entry = self.log.write_entry(self.model)
        self.tensorboard_log.write_entry(self.model)
        print(entry)

    def _optimize(self):
        numiter = self.flags.test_every
        Loop(self.optimizers, stop=numiter)()
        # self.optimizer.minimize(self.model, maxiter=numiter, global_step=self.global_step)

    def _setup_model(self):
        filter_size = 5
        patch_length = filter_size**2
        conv_features = PatchInducingFeature.from_images(self.X_train.reshape(-1, 28, 28), self.flags.M,
                filter_size)
        h1_out = 576
        conv_mean = Conv2dMean(filter_size, 1)
        conv_mean.set_trainable(False)

        sess = conv_mean.enquire_session()
        H1_X = sess.run(conv_mean(self.X_train.reshape(-1, 28, 28, 1)))
        Z_rbf = select_initial_inducing_points(H1_X, self.flags.M)
        rbf_features = features.InducingPoints(Z_rbf)

        layers = [
                ConvLayer(
                    base_kernel=kernels.RBF(patch_length),
                    mean_function=conv_mean,
                    feature=conv_features,
                    input_size=(28, 28),
                    feature_maps=1,
                    filter_size=filter_size,
                    white=False),
                SVGP_Layer(gpflow.kernels.RBF(h1_out, lengthscales=2, variance=2), num_outputs=10,
                    feature=rbf_features,
                    mean_function=gpflow.mean_functions.Zero(output_dim=10),
                    white=False)
        ]

        self.model = DGP_Base(self.X_train, self.Y_train,
                likelihood=gpflow.likelihoods.MultiClass(10),
                layers=layers,
                minibatch_size=self.flags.batch_size)
        print("z initialized")
        self.model.compile()

    def _setup_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.model.enquire_session().run(self.global_step.initializer)

        self.optimizers = []
        if self.flags.optimizer == "NatGrad":
            variational_parameters = [[self.model.layers[0].M1_q_mu, self.model.layers[0].IMM_q_sqrt],
                    [self.model.layers[1].q_mu, self.model.layers[1].q_sqrt]]

            for params in variational_parameters:
                for param in params:
                    param.set_trainable(False)

            nat_grad = gpflow.train.NatGradOptimizer(gamma=0.001).make_optimize_action(self.model,
                    var_list=variational_parameters)
            self.optimizers.append(nat_grad)

        if self.flags.optimizer == "SGD":
            opt = gpflow.train.GradientDescentOptimizer(learning_rate=self.flags.lr)\
                    .make_optimize_action(self.model, global_step=self.global_step)
        elif self.flags.optimizer == "Adam":
            opt = gpflow.train.AdamOptimizer(learning_rate=self.flags.lr).make_optimize_action(self.model,
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
        sample_task = utils.LayerOutputLogger(self.model)
        model_parameter_task = utils.ModelParameterLogger(self.model)
        likelihood = utils.LogLikelihoodLogger()

        tasks = [sample_task, model_parameter_task, likelihood]
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
    parser.add_argument('-M', type=int, default=64,
            help="How many inducing points to use.")
    parser.add_argument('--name', type=str, required=True,
            help="What to call the experiment. Determines the directory where results are dumped.")
    parser.add_argument('--lr-decay-steps', type=int, default=50000,
            help="The program uses exponential learning rate decay with 0.1 decay every lr-decay-steps.")
    parser.add_argument('--test-every', type=int, default=5000,
            help="How often to evaluate the test accuracy. Unit optimization iterations.")
    parser.add_argument('--test-size', type=int, default=10000)
    parser.add_argument('--random-inducing', action='store_true', default=False)
    parser.add_argument('--log-dir', type=str, default='results',
            help="Directory to write the results to.")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128,
            help="Minibatch size to use in optimization.")
    parser.add_argument('--tensorboard-dir', type=str, default='/tmp/mnist/tensorboard')
    parser.add_argument('--optimizer', type=str, default='Adam',
            help="Either Adam or NatGrad")
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


