import numpy as np
import tensorflow as tf
import gpflow
import observations
import utils
from gpflow import settings
from sklearn import preprocessing
from arguments import default_parser, train_steps
from experiment import Experiment

class Cifar(Experiment):
    def _load_data(self):
        (X_train, Y_train), (X_test, Y_test) = observations.cifar10('/tmp/cifar10')
        X_train = np.transpose(X_train, [0, 2, 3, 1]).astype(settings.float_type)
        X_test = np.transpose(X_test, [0, 2, 3, 1]).astype(settings.float_type)
        Y_train = Y_train.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)

        X_test = np.concatenate([X_train[self.flags.N:], X_test], axis=0)
        Y_test = np.concatenate([Y_train[self.flags.N:], Y_test], axis=0)
        X_train = X_train[0:self.flags.N]
        Y_train = Y_train[0:self.flags.N]

        assert(Y_train.shape[0] == self.flags.N)
        assert(X_train.shape[0] == self.flags.N)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self._preprocess_data()

    def _preprocess_data(self):
        mean = self.X_train.mean(axis=(0, 1, 2))
        self.X_train -= mean
        self.X_test -= mean
        std = self.X_train.std(axis=(0, 1, 2))
        self.X_train = self.X_train / std
        self.X_test = self.X_test / std

def read_args():
    parser = default_parser()
    parser.add_argument('--tensorboard-dir', type=str, default='/tmp/cifar10/tensorboard')
    parser.add_argument('-N', type=int, default=50000, help="Use N training examples.")
    return parser.parse_args()

def main():
    flags = read_args()

    experiment = Cifar(flags)
    try:
        for i in range(train_steps(flags)):
            experiment.train_step()
    finally:
        experiment.conclude()

if __name__ == "__main__":
    main()
