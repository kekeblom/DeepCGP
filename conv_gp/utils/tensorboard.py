import io
import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
import math
from doubly_stochastic_dgp.layers import SVGP_Layer
from layers import ConvLayer
from .log import LogBase

class TensorBoardTask(object):
    def __call__(self, model):
        return model.enquire_session().run(self.summary)

class LogLikelihoodLogger(TensorBoardTask):
    def __init__(self):
        self.title = 'train_log_likelihood'
        self.batch_size = 64
        self.likelihood_holder = tf.placeholder(settings.float_type, shape=())
        self.summary = tf.summary.scalar(self.title, self.likelihood_holder)

    def __call__(self, model):
        with gpflow.decors.params_as_tensors_for(model):
            X_holder, Y_holder = model.X, model.Y
        log_likelihood = 0.0
        compute_on = min(5000, model.X._value.shape[0])
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

        log_likelihood = log_likelihood / (batches * self.batch_size)

        return model.enquire_session().run(self.summary, feed_dict={
            self.likelihood_holder: log_likelihood
        })

class LayerOutputLogger(TensorBoardTask):
    def __init__(self, model, X_test):
        self.X_test = X_test
        self.input_image = tf.placeholder(settings.float_type, shape=[None, None])
        self.summary = self._build_summary()
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        self.plt = pyplot

    def _build_summary(self):
        self.tf_sample_image = tf.placeholder(settings.float_type, shape=[None] * 4)
        self.tf_mean_image = tf.placeholder(settings.float_type, shape=[None] * 4)
        self.tf_variance_image = tf.placeholder(settings.float_type, shape=[None] * 4)
        summaries = [
                tf.summary.image("conv_sample", self.tf_sample_image),
                tf.summary.image("conv_mean", self.tf_mean_image),
                tf.summary.image("conv_var", self.tf_variance_image)
                ]
        return tf.summary.merge(summaries)

    def __call__(self, model):
        chosen = np.random.choice(np.arange(len(self.X_test)), size=1)

        conv_layer = model.layers[0]

        sess = model.enquire_session()

        with gpflow.params_as_tensors_for(conv_layer):
            samples, Fmeans, Fvars = conv_layer.sample_from_conditional(
                    tf.tile(self.input_image[None], [4, 1, 1]), full_cov=False)
            samples, Fmeans, Fvars = sess.run([samples, Fmeans, Fvars], {
                self.input_image: self.X_test[chosen]
                })

        sample_image = self._plot_samples(samples[:, 0, :], conv_layer)
        mean_image = self._plot_mean(Fmeans[:, 0, :], conv_layer)
        variance_image = self._plot_variance(Fvars[:, 0, :], conv_layer)

        sample_image, mean_image, variance_image = sess.run([
            sample_image, mean_image, variance_image])

        self.plt.close('all')

        return sess.run(self.summary, {
            self.tf_sample_image: sample_image,
            self.tf_mean_image: mean_image,
            self.tf_variance_image: variance_image
            })

    def _plot_samples(self, samples, conv_layer):
        sample_count = len(samples)
        feature_maps = conv_layer.gp_count
        sample_figure = self.plt.figure(figsize=(feature_maps * 5, sample_count * 5))
        height_width = int(np.sqrt(samples.shape[1] / feature_maps))
        samples = samples.reshape(sample_count, height_width, height_width, feature_maps)
        samples = np.transpose(samples, [0, 3, 1, 2])

        for sample in range(sample_count):
            for feature_map in range(feature_maps):
                axis = self.plt.subplot2grid((sample_count, feature_maps), loc=(sample, feature_map))
                axis.set_title("F sample {} feature map {}".format(sample, feature_map))
                image = samples[sample, feature_map, :, :]
                img = axis.imshow(image)
        sample_figure.colorbar(img, ax=sample_figure.axes)
        return self._figure_to_tensor(sample_figure)

    def _plot_mean(self, Fmeans, conv_layer):
        feature_maps = conv_layer.gp_count
        mean_figure = self.plt.figure(figsize=(5 * feature_maps, 5))
        image = Fmeans[0]
        height = int(np.sqrt(image.size / feature_maps))
        image = image.reshape(height, height, feature_maps)
        image = np.transpose(image, [2, 0, 1])
        for i in range(feature_maps):
            axis = self.plt.subplot2grid((1, feature_maps), loc=(0, i))
            axis.set_title("Mean fm {}".format(i))
            img = axis.imshow(image[i])

        mean_figure.colorbar(img, ax=mean_figure.axes)
        return self._figure_to_tensor(mean_figure)

    def _plot_variance(self, Fvars, conv_layer):
        feature_maps = conv_layer.gp_count
        variance_figure = self.plt.figure(figsize=(5 * feature_maps, 5))
        image = Fvars[0]
        height = int(np.sqrt(image.size / feature_maps))
        image = image.reshape(height, height, feature_maps)
        image = np.transpose(image, [2, 0, 1])
        for i in range(feature_maps):
            axis = self.plt.subplot2grid((1, feature_maps), loc=(0, i))
            img = axis.imshow(image[i])
            axis.set_title("Variance fm {}".format(i))
        variance_figure.colorbar(img, ax=variance_figure.axes)
        return self._figure_to_tensor(variance_figure)

    def _figure_to_tensor(self, figure):
        byte_buffer = io.BytesIO()
        figure.savefig(byte_buffer, format='png')
        byte_buffer.seek(0)
        image = tf.image.decode_png(byte_buffer.getvalue(), channels=4)
        return tf.expand_dims(image, 0) # Add batch dimension.


class ModelParameterLogger(TensorBoardTask):
    def __init__(self, model):
        self.summary = self._build_summary(model)

    def _build_summary(self, model):
        parameters = list(model.parameters)
        scalar_params = [p for p in parameters if p.size == 1]
        other_params = [p for p in parameters if p.size > 1]

        scalar_summaries = [tf.summary.scalar(p.pathname, tf.reshape(p.constrained_tensor, []))
                for p in scalar_params]
        other_summaries = [tf.summary.histogram(p.full_name, p.constrained_tensor)
                for p in other_params]

        return tf.summary.merge(scalar_summaries + other_summaries)

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


