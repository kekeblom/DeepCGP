import numpy as np
import tensorflow as tf
import gpflow
import observations
from gpflow import settings
from kernels import ConvKernel, PatchInducingFeature

class MNIST(object):
    def __init__(self):
        self._load_data()
        self._setup_model()
        self._setup_optimizer()
        self.epoch = 0

    def train_epoch(self):
        self._step()
        self._optimize()
        self._evaluate()

    def _optimize(self):
        maxiter = self.X_train.shape[0] // self.minibatch_size
        self.optimizer.minimize(self.model, maxiter=maxiter, global_step=self.global_step)
        self._print_status_line()

    def _step(self):
        self.epoch += 1

    def _print_status_line(self):
        sess = self.model.enquire_session()
        objective = sess.run(self.model.objective)
        print("Epoch: {}, objective: {}".format(self.epoch, objective))
        lr = self.model.enquire_session().run(self.lr_schedule)
        print('lr: {}'.format(lr))

    def _evaluate(self):
        correct = 0
        batch_size = 512
        for i in range(len(self.Y_test) // batch_size + 1):
            the_slice = slice(i * batch_size, (i+1) * batch_size)
            X = self.X_test[the_slice]
            Y = self.Y_test[the_slice]
            probabilities, _ = self.model.predict_y(X)
            predicted_class = probabilities.argmax(axis=1)
            correct += (predicted_class == Y).sum()
        print("Accuracy: {}".format(correct / self.Y_test.size))

    def _setup_model(self):
        self.num_inducing = 750
        window_size = 5
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
        self.minibatch_size = 128
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
        decay_every = 10
        self.lr_schedule = tf.train.exponential_decay(1e-2, self.global_step,
                decay_steps=len(self.X_train) // self.minibatch_size * decay_every,
                decay_rate=0.1)
        self.optimizer = gpflow.train.AdamOptimizer(learning_rate=self.lr_schedule)

    def _load_data(self):
        (self.X_train, self.Y_train), (
                self.X_test, self.Y_test) = observations.mnist('/tmp/mnist/')
        self.X_train = self._preprocess(self.X_train)
        self.X_test = self._preprocess(self.X_test)

    def _preprocess(self, data):
        return (data / 255.0).astype(settings.float_type)


if __name__ == "__main__":
    experiment = MNIST()
    while True:
        experiment.train_epoch()


