import math
import numpy as np
import tensorflow as tf
import gpflow
import observations
import utils
from gpflow import settings
from kernels import ConvKernel, PatchInducingFeature

class MNIST(object):
    def __init__(self, flags):
        self.flags = flags

        self._load_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logger()
        self._write_initial_inducing_points()

    def conclude(self):
        self.log.write_model(self.model)
        self.log.close()

    def train_step(self):
        self._optimize()
        self._log_step()

    def _log_step(self):
        entry = self.log.write_entry(self.model)
        print(entry)

    def _optimize(self):
        numiter = self.flags.test_every
        self.optimizer.minimize(self.model, maxiter=numiter, global_step=self.global_step)

    def _setup_model(self):
        num_inducing = self.flags.M
        filter_size = 5
        patch_size = filter_size**2
        kernel = ConvKernel(
                gpflow.kernels.RBF(input_dim=patch_size),
                image_size=(28, 28),
                filter_size=filter_size,
                stride=1,
                channels=1)
        Z = kernel.init_inducing_patches(self.X_train, num_inducing)
        inducing_features = PatchInducingFeature(Z)
        print("z initialized")
        self.minibatch_size = self.flags.batch_size
        self.model = gpflow.models.SVGP(self.X_train, self.Y_train,
                kern=kernel,
                feat=inducing_features,
                num_latent=10,
                likelihood=gpflow.likelihoods.MultiClass(10),
                whiten=False,
                minibatch_size=self.minibatch_size)
        self.model.compile()

    def _setup_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.model.enquire_session().run(self.global_step.initializer)
        self.lr_schedule = tf.train.exponential_decay(self.flags.lr, self.global_step,
                decay_steps=self.flags.lr_decay_steps,
                decay_rate=0.1)
        self.optimizer = gpflow.train.AdamOptimizer(learning_rate=self.lr_schedule)

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
        self.X_train = self._preprocess(self.X_train)
        self.X_test = self._preprocess(self.X_test)
        self.Y_train, self.Y_test

    def _select_training_points(self):
        chosen = slice(0, self.flags.N)
        self.X_train = self.X_train[chosen, :]
        self.Y_train = self.Y_train[chosen, :]

    def _preprocess(self, data):
        return (data / 255.0).astype(settings.float_type)

    def _setup_logger(self):
        loggers = [
            utils.GlobalStepLogger(),
            utils.LearningRateLogger(self.lr_schedule),
            utils.AccuracyLogger(self.X_test, self.Y_test),
            utils.LogLikelihoodLogger()
        ]
        self.log = utils.Log(self.flags.log_dir, self.flags.name, loggers)
        self.log.write_flags(self.flags)

    def _write_initial_inducing_points(self):
        self.log.write_inducing_points(self.model, "z_init.npy")


def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fashion', action='store_true', default=False,
            help="Use fashion MNIST instead of regular MNIST.")
    parser.add_argument('-N', type=int,
            help="How many training examples to use.", default=60000)
    parser.add_argument('-M', type=int,
            help="How many inducing points to use.")
    parser.add_argument('--name', type=str, required=True,
            help="What to call the experiment. Determines the directory where results are dumped.")
    parser.add_argument('--lr-decay-steps', type=int, default=100000,
            help="The program uses exponential learning rate decay with 0.1 decay every lr-decay-steps.")
    parser.add_argument('--test-every', type=int, default=10000,
            help="How often to evaluate the test accuracy. Unit optimization iterations.")
    parser.add_argument('--log-dir', type=str, default='results',
            help="Directory to write the results to.")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128,
            help="Minibatch size to use in optimization.")
    return parser.parse_args()

def train_steps(flags):
    # Roughly until the learning rate becomes 1e-5
    decay_count = math.log(1e-5 / flags.lr, 0.1) # How many times decay has to be applied to reach 1e-5.
    return math.ceil(flags.lr_decay_steps * decay_count / flags.test_every)

def main():
    np.random.seed(5033)

    flags = read_args()

    experiment = MNIST(flags)

    try:
        for i in range(train_steps(flags)):
            experiment.train_step()
    finally:
        experiment.conclude()

if __name__ == "__main__":
    main()


