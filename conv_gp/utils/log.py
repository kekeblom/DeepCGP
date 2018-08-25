import csv
import os
import gpflow
import toml
import numpy as np
import tensorflow as tf
from gpflow import settings

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

    def __call__(self, model):
        correct = 0
        batch_size = 32
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
        return correct / self.Y_test.size

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
        self.file = open(file_path, 'at')
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

