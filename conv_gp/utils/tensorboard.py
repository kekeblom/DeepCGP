import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
import math
from doubly_stochastic_dgp.layers import SVGP_Layer
from .log import LogBase

class TensorBoardTask(object):
    def __call__(self, model):
        return model.enquire_session().run(self.summary)

class LogLikelihoodLogger(TensorBoardTask):
    def __init__(self):
        self.title = 'train_log_likelihood'
        self.batch_size = 512
        self.likelihood_holder = tf.placeholder(settings.float_type, shape=())
        self.summary = tf.summary.scalar(self.title, self.likelihood_holder)

    def __call__(self, model):
        with gpflow.decors.params_as_tensors_for(model):
            X_holder, Y_holder = model.X, model.Y
        log_likelihood = 0.0
        compute_on = 5000
        batches = math.ceil(compute_on / self.batch_size)
        for i in range(batches):
            the_slice = slice(i * self.batch_size, (i+1) * self.batch_size)
            X = model.X._value[the_slice]
            Y = model.Y._value[the_slice]
            batch_likelihood = model.compute_log_likelihood(feed_dict={
                X_holder: X,
                Y_holder: Y
                })
            log_likelihood += batch_likelihood

        return model.enquire_session().run(self.summary, feed_dict={
            self.likelihood_holder: log_likelihood
        })

class LayerOutputLogger(TensorBoardTask):
    def __init__(self, model):
        self.summary = self._build_summary(model)

    def _build_summary(self, model):
        self.input_image = tf.placeholder(settings.float_type, shape=[None, 784])

        Fs, Fmeans, _ = model.propagate(self.input_image)
        side = int(np.sqrt(model.layers[0].view.patch_count))
        mean_image = tf.reshape(Fmeans[0], [-1, side, side, 1])
        sample_image = tf.reshape(Fs[0], [-1, side, side, 1])

        input_sum = tf.summary.image("conv_input_image", tf.reshape(self.input_image, [-1, 28, 28, 1]))
        sample_sum = tf.summary.image("conv_sample", sample_image)
        mean_sum = tf.summary.image("conv_mean", mean_image)

        return tf.summary.merge([input_sum, sample_sum, mean_sum])

    def __call__(self, model):
        X = model.X.value
        samples = 3
        random_indices = np.random.randint(X.shape[0], size=samples)
        x = X[random_indices]

        return model.enquire_session().run(self.summary, {
            self.input_image: x
        })

class ModelParameterLogger(TensorBoardTask):
    def __init__(self, model):
        self.summary = self._build_summary(model)

    def _build_summary(self, model):
        # Variational distribution parameters.
        q_mu = model.layers[0].q_mu.parameter_tensor
        q_sqrt = model.layers[0].q_sqrt.parameter_tensor
        q_mu_sum = tf.summary.histogram('q_mu', q_mu)
        q_sqrt_sum = tf.summary.histogram('q_sqrt', q_sqrt)

        # Inducing points.
        conv_layer = model.layers[0]
        Z = conv_layer.feature.Z.parameter_tensor
        Z_shape = tf.shape(Z)
        Z_sum = tf.summary.histogram('Z', Z)

        variance = conv_layer.base_kernel.variance.parameter_tensor
        length_scale = conv_layer.base_kernel.lengthscales.parameter_tensor

        var_sum = tf.summary.scalar('base_kernel_var', variance)
        ls_sum = tf.summary.scalar('base_kernel_length_scale', length_scale)

        rbf_layer = [layer for layer in model.layers if isinstance(layer, SVGP_Layer)][0]
        rbf_var_sum = tf.summary.histogram('rbf_var',
                rbf_layer.kern.variance.parameter_tensor)
        rbf_ls_sum = tf.summary.histogram('rbf_lengthscale',
                rbf_layer.kern.lengthscales.parameter_tensor)

        return tf.summary.merge([
            q_mu_sum,
            q_sqrt_sum,
            Z_sum,
            var_sum,
            ls_sum,
            rbf_var_sum,
            rbf_ls_sum])

class PatchCovarianceLogger(TensorBoardTask):
    def __init__(self, model):
        self.covariance_holder = tf.placeholder(settings.float_type, [None] * 4)
        self.image_holder = tf.placeholder(settings.float_type, [None] * 4)
        covariance = tf.summary.image('Kuf_covariance', self.covariance_holder)
        patches = tf.summary.image('Kuf_image', self.image_holder)
        self.summary = tf.summary.merge([covariance, patches])

    def __call__(self, model):
        conv_layer = model.layers[0]
        Z = conv_layer.feature.Z.value

        sess = model.enquire_session()
        X = model.X.value
        chosen = np.random.choice(np.arange(len(X)), size=1)
        X = X[chosen]
        view = conv_layer.view
        patches = view.extract_patches_PNL(X.reshape(-1, view.input_size[0],
            view.input_size[1], view.feature_maps))

        covariance = conv_layer.conv_kernel.Kuf(Z, patches)
        covariance = sess.run(tf.transpose(covariance, [2, 0, 1]))[:, :, :, None]

        patch_height = conv_layer.view.patch_shape[0]
        patch_width = conv_layer.view.patch_shape[1]
        image = X.reshape([-1, *conv_layer.view.input_size,
            conv_layer.view.feature_maps])

        return sess.run(self.summary, {
            self.covariance_holder: covariance,
            self.image_holder: image
        })


class TensorBoardLog(LogBase):
    def __init__(self, tasks, tensorboard_dir, name, model, global_step):
        self.global_step = global_step
        log_dir = self._log_dir(tensorboard_dir, name)
        self.writer = tf.summary.FileWriter(log_dir, model.enquire_graph())
        self.tasks = tasks

    def write_entry(self, model):
        sess = model.enquire_session()
        summaries = [task(model) for task in self.tasks]
        step = sess.run(self.global_step)
        for summary in summaries:
            self.writer.add_summary(summary, global_step=step)


