import os
import numpy as np
import tensorflow as tf
import gpflow
import observations
import utils
from gpflow import settings
from gpflow.actions import Loop
from models import build_model

class Experiment(object):
    def __init__(self, flags):
        self.flags = flags

        self._load_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logger()
        if flags.load_model:
            self._load_model()

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

    def _optimize(self):
        numiter = self.flags.test_every
        Loop(self.optimizers, stop=numiter)()

    def _model_path(self):
        return os.path.join(self.flags.log_dir, self.flags.name + '.npy')

    def _save_model_parameters(self):
        trainables = self.model.read_trainables()
        trainables['global_step'] = self.model.enquire_session().run(self.global_step)
        np.save(self._model_path(), trainables)

    def _load_model(self):
        trainables = np.load(self._model_path()).item()
        global_step = trainables['global_step']
        del trainables['global_step']
        self.model.assign(trainables)
        sess = self.model.enquire_session()
        sess.run(self.global_step.assign(global_step))

    def _setup_model(self):
        self.model = build_model(self.flags, self.X_train, self.Y_train)

    def _setup_learning_rate(self):
        self.learning_rate = tf.train.exponential_decay(self.flags.lr, global_step=self.global_step,
                decay_rate=0.1, decay_steps=self.flags.lr_decay_steps)

    def _setup_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        self._setup_learning_rate()
        self.model.enquire_session().run(self.global_step.initializer)

        self.optimizers = []
        if self.flags.optimizer == "NatGrad":
            variational_parameters = [(self.model.layers[-1].q_mu, self.model.layers[-1].q_sqrt)]

            for params in variational_parameters:
                for param in params:
                    param.set_trainable(False)

            nat_grad = gpflow.train.NatGradOptimizer(gamma=self.flags.gamma).make_optimize_action(self.model,
                    var_list=variational_parameters)
            self.optimizers.append(nat_grad)

        if self.flags.optimizer == "SGD":
            opt = gpflow.train.GradientDescentOptimizer(learning_rate=self.learning_rate)\
                    .make_optimize_action(self.model, global_step=self.global_step)
            self.optimizers.append(opt)
        elif self.flags.optimizer == "Adam" or self.flags.optimizer == "NatGrad":
            opt = gpflow.train.AdamOptimizer(learning_rate=self.learning_rate).make_optimize_action(self.model,
                    global_step=self.global_step)
            self.optimizers.append(opt)

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


