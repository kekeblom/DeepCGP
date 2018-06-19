import numpy as np
import tensorflow as tf
import gpflow
import observations
from gpflow import settings
from kernels import ConvKernel, PatchInducingFeature

class Classification(object):
    def __init__(self):
        self._load_data()
        self.num_inducing = 25
        window_size = 3
        patch_size = window_size**2
        kernel = ConvKernel(
                gpflow.kernels.RBF(input_dim=patch_size),
                image_size=(28, 28),
                window_size=window_size,
                stride=1,
                channels=1)
        Z = kernel.init_inducing_patches(self.X_train, self.num_inducing)
        inducing_features = PatchInducingFeature(Z)
        print("z initialized")
        self.model = gpflow.models.SVGP(self.X_train, self.Y_train,
                kern=kernel,
                feat=inducing_features,
                num_latent=10,
                likelihood=gpflow.likelihoods.MultiClass(10),
                whiten=True,
                minibatch_size=32)
        self.model.compile()


    def run(self):
        optimizer = gpflow.train.GradientDescentOptimizer(learning_rate=0.01)
        optimizer.minimize(self.model)
        print(self.model)

    def _load_data(self):
        (self.X_train, self.Y_train), (
                self.X_test, self.Y_test) = observations.mnist('/tmp/mnist/')
        def reshape(X):
            return X.reshape(-1, 28, 28, 1)
        self.X_train = reshape(self.X_train)[0:10000].astype(settings.float_type)
        self.X_test = reshape(self.X_test).astype(settings.float_type)


if __name__ == "__main__":
    experiment = Classification()
    experiment.run()


