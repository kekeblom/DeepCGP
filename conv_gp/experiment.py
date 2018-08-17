import os
import numpy as np
import tensorflow as tf
import gpflow
import observations
import utils
from gpflow import settings
from gpflow.actions import Loop
from models import ModelBuilder

float_type = settings.float_type

class Experiment(object):
    def __init__(self, flags):
        self.flags = flags

        self._load_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logger()

    def _load_data(self):
        raise NotImplementedError()

    def conclude(self):
        self.log.close()

    def train_step(self):
        self._optimize()
        self._log_step()
        self._save_model_parameters()

    def _log_step(self):
        entry = self.log.write_entry(self.model)
        self.tensorboard_log.write_entry(self.model)
        print(entry)

    def _optimize(self, retry=0, error=None):
        numiter = self.flags.test_every
        max_retries = 5
        if retry > max_retries:
            raise error
        try:
            Loop(self.loop, stop=numiter)()
        except tf.errors.InvalidArgumentError as exception:
            if self.flags.optimizer != "NatGrad":
                raise exception
            self.step_back_gamma()
            self._optimize(retry=retry+1, error=exception)

    def _model_path(self, model_name=None):
        if model_name is None:
            model_name = self.flags.name
        return os.path.join(self.flags.log_dir, model_name + '.npy')

    def _save_model_parameters(self):
        params = {}
        sess = self.model.enquire_session()
        for param in self.model.parameters:
            value = sess.run(param.constrained_tensor)
            key = param.pathname
            params[key] = value
        params['global_step'] = sess.run(self.global_step)
        np.save(self._model_path(), params)

    def _setup_model(self):
        model_builder = ModelBuilder(self.flags,
                self.X_train, self.Y_train, model_path=self._model_path(self.flags.load_model))
        self.model = model_builder.build()

    def _setup_learning_rate(self):
        self.learning_rate = tf.train.exponential_decay(self.flags.lr, global_step=self.global_step,
                decay_rate=0.1, decay_steps=self.flags.lr_decay_steps, staircase=True)
        gamma_max = 1.0
        gamma_step = 1e-3
        back_step = tf.constant(0.2, dtype=float_type)
        t = tf.cast(self.global_step, dtype=float_type) / 100.0
        steps_back = tf.Variable(0.0, dtype=float_type)
        self.gamma = tf.minimum((t * gamma_step + self.flags.gamma) * tf.pow(back_step, steps_back), gamma_max)
        self.step_back_gamma = utils.RunOpAction(steps_back.assign(steps_back + 1.0))

        self.model.enquire_session().run(tf.variables_initializer([steps_back]))

    def _setup_optimizer(self):
        self.loop = []
        self.global_step = tf.train.get_or_create_global_step()
        self._setup_learning_rate()
        self.model.enquire_session().run(self.global_step.initializer)

        if self.flags.optimizer == "NatGrad":
            variational_parameters = [(l.q_mu, l.q_sqrt) for l in self.model.layers]

            for params in variational_parameters:
                for param in params:
                    param.set_trainable(False)

            nat_grad = gpflow.train.NatGradOptimizer(gamma=self.gamma).make_optimize_action(self.model,
                    var_list=variational_parameters)
            self.loop.append(nat_grad)

        if self.flags.optimizer == "SGD":
            opt = gpflow.train.GradientDescentOptimizer(learning_rate=self.learning_rate)\
                    .make_optimize_action(self.model, global_step=self.global_step)
            self.loop.append(opt)
        elif self.flags.optimizer == "Adam" or self.flags.optimizer == "NatGrad":
            opt = gpflow.train.AdamOptimizer(learning_rate=self.learning_rate).make_optimize_action(self.model,
                    global_step=self.global_step)
            self.loop.append(opt)

        if self.flags.optimizer not in ["Adam", "NatGrad", "SGD"]:
            raise ValueError("Not a supported optimizer. Try Adam or NatGrad.")

    def _setup_logger(self):
        self._init_logger()
        self._init_tensorboard()

    def _init_logger(self):
        X_test = self.X_test.reshape(self.X_test.shape[0], -1)
        loggers = [
            utils.GlobalStepLogger(),
            utils.AccuracyLogger(X_test, self.Y_test),
        ]
        self.log = utils.Log(self.flags.log_dir,
                self.flags.name,
                loggers)
        self.log.write_flags(self.flags)

    def _init_tensorboard(self):
        X_test = self.X_test.reshape(self.X_test.shape[0], -1)
        sample_task = utils.LayerOutputLogger(self.model, X_test)
        model_parameter_task = utils.ModelParameterLogger(self.model)
        likelihood = utils.LogLikelihoodLogger()

        tasks = [sample_task, likelihood, model_parameter_task]
        self.tensorboard_log = utils.TensorBoardLog(tasks, self.flags.tensorboard_dir, self.flags.name,
                self.model, self.global_step)


