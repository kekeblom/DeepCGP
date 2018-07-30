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
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = observations.cifar10('/tmp/cifar10')
        self.X_train = np.transpose(self.X_train, [0, 2, 3, 1]).astype(settings.float_type)
        self.X_test = np.transpose(self.X_test, [0, 2, 3, 1]).astype(settings.float_type)
        self.Y_train = self.Y_train.reshape(-1, 1)
        self.Y_test = self.Y_test.reshape(-1, 1)
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
    parser.add_argument('--fashion', action='store_true', default=False,
            help="Use fashion MNIST instead of regular MNIST.")
    parser.add_argument('--tensorboard-dir', type=str, default='/tmp/cifar10/tensorboard')
    parser.add_argument('--model-path', default='/tmp/cifar10/model.npy')
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
