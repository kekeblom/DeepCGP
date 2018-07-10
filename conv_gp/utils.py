import csv
import math
import os
import gpflow
import toml
import numpy as np
import tensorflow as tf

def ensure_dir(path):
    if not os.path.exists(path):
        all_but_last = os.path.split(path)[0:-1]
        all_but_last = [p for p in all_but_last if p]
        if len(all_but_last) > 0:
            ensure_dir(os.path.join(*all_but_last))
        os.mkdir(path)

class Logger(object):
    """Logger is an abstract base class for loggers.
    Loggers can be passed to the log. At each log write each logger will be called
    and the output will be logged.

    The property title should be set by subclasses. It will determine the column title.
    """

    def __call__(self, model):
        """implemented by subclass. Should return whatever wants to be logged."""
        raise NotImplementedError()

class GlobalStepLogger(Logger):
    def __init__(self):
        self.title = "global_step"

    def __call__(self, model):
        sess = model.enquire_session()
        global_step = tf.train.get_or_create_global_step()
        return sess.run(global_step)

class LearningRateLogger(Logger):
    def __init__(self, learning_rate_op):
        self.title = "lr"
        self.learning_rate_op = learning_rate_op

    def __call__(self, model):
        sess = model.enquire_session()
        return sess.run(self.learning_rate_op)

    def tensorboard_op(self, model):
        tf.summary.scalar(self.title, self.learning_rate_op)

class AccuracyLogger(Logger):
    def __init__(self, X_test, Y_test):
        self.title = 'test_accuracy'
        self.X_test, self.Y_test = X_test, Y_test
        self.prev_accuracy = None

    def __call__(self, model):
        correct = 0
        batch_size = 64
        for i in range(len(self.Y_test) // batch_size + 1):
            the_slice = slice(i * batch_size, (i+1) * batch_size)
            X = self.X_test[the_slice]
            Y = self.Y_test[the_slice]
            mean_samples, _ = model.predict_y(X, 10)
            # Grab the mean probability over all samples.
            # Then argmax to get the final prediction.
            probabilities = mean_samples.mean(axis=0)
            predicted_class = probabilities.argmax(axis=1)[:, None]
            correct += (predicted_class == Y).sum()
        accuracy = correct / self.Y_test.size
        self.prev_accuracy = accuracy
        return accuracy

class LogLikelihoodLogger(Logger):
    def __init__(self):
        self.title = 'train_log_likelihood'
        self.batch_size = 512

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
        return log_likelihood

class LayerOutputLogger(object):
    def __init__(self, model):
        self.summary = self._build_summary(model)

    def _build_summary(self, model):
        X = model.X.parameter_tensor
        samples = 10
        random_indices = tf.random_uniform([10], 0, tf.shape(X)[0], dtype=tf.int32)
        x = tf.gather(X, random_indices, axis=0)

        input_image = tf.reshape(x, [samples, 28, 28, 1])
        input_sum = tf.summary.image("conv_input_image", input_image)

        Fs, Fmeans, _ = model.propagate(x)
        sample_image = tf.reshape(Fs[0], [samples, 24, 24, 1])
        sample_sum = tf.summary.image("conv_sample", sample_image)
        mean_image = tf.reshape(Fmeans[0], [samples, 24, 24, 1])
        mean_sum = tf.summary.image("conv_mean", mean_image)
        return tf.summary.merge([input_sum, sample_sum, mean_sum])

class ModelParameterLogger(object):
    def __init__(self, model):
        self.summary = self._build_summary(model)

    def _build_summary(self, model):
        # Variational distribution parameters.
        q_mu = model.layers[0].M1_q_mu.read_value()
        q_sqrt = model.layers[0].IMM_q_sqrt.read_value()[0, :, :]
        q_mu_sum = tf.summary.tensor_summary('q_mu', q_mu)
        q_sqrt_sum = tf.summary.tensor_summary('q_sqrt', q_sqrt)

        # Inducing points.
        Z = model.layers[0].feature.Z.read_value()
        Z_shape = tf.shape(Z)
        Z_sum = tf.summary.image('Z', tf.reshape(Z, [Z_shape[0], 5, 5, 1]))
        return tf.summary.merge([q_mu_sum, q_sqrt_sum, Z_sum])

class ModelSaver(object):
    def __init__(self, model, test_dir):
        self.model = model
        self.test_dir = test_dir

    def save(self):
        self._save_model()

    def _save_model(self):
        saver = tf.train.Saver()

        path = os.path.join(self.test_dir, "model.ckpt")
        sess = self.model.enquire_session()
        saver.save(sess, path)

class LogBase(object):
    def _log_dir(self, log_dir, run_name):
        path = os.path.join(log_dir, run_name)
        ensure_dir(path)
        return path

class TensorBoardLog(LogBase):
    def __init__(self, tasks, tensorboard_dir, name, model):
        log_dir = self._log_dir(tensorboard_dir, name)
        self.writer = tf.summary.FileWriter(log_dir, model.enquire_graph())
        self._collect_summary(tasks)

    def _collect_summary(self, tasks):
        summaries = [task.summary for task in tasks]
        self.summary_op = tf.summary.merge(summaries)

    def write_entry(self, model):
        summary = model.enquire_session().run(self.summary_op)
        self.writer.add_summary(summary)

class Log(LogBase):
    def __init__(self, log_dir, run_name, loggers):
        self.loggers = loggers
        self.log_dir = self._log_dir(log_dir, run_name)
        self._start_log_file(run_name)
        self._write_headers()
        self.entries = 0

    def _write_headers(self):
        self.headers = ["Entry"] + [l.title for l in self.loggers]
        self.csv_writer.writerow(self.headers)

    def _start_log_file(self, name):
        file_path = os.path.join(self.log_dir, 'log.csv')
        self.file = open(file_path, 'wt')
        self.csv_writer = csv.writer(self.file)

    def _human_readable(self, entry):
        abuffer = []
        for key, value in zip(self.headers, entry):
            abuffer.append("{key}: {value}".format(key=key, value=value))
        return "; ".join(abuffer)

    def write_entry(self, model):
        entry = [self.entries] + [logger(model) for logger in self.loggers]
        self.csv_writer.writerow(entry)
        self.entries += 1
        return self._human_readable(entry)

    def write_flags(self, flags):
        flags = vars(flags) # As dictionary.
        arg_file = os.path.join(self.log_dir, 'options.toml')
        with open(arg_file, 'wt') as f:
            toml.dump(flags, f)

    def write_model(self, model):
        saver = ModelSaver(model, self.log_dir)
        saver.save()
        self.write_inducing_points(model, "inducing_points.npy")

    def write_inducing_points(self, model, filename):
        path = os.path.join(self.log_dir, filename)
        np.save(path, model.feature.Z._value)

    def close(self):
        self.file.close()

