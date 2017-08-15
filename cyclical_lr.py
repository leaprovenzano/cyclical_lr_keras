from collections import defaultdict

import numpy as np

from keras import callbacks
from keras import backend as K


class CycleFunctionUnavailable(Exception):
    """Error CyclicLr callback func arg"""


class CyclicLR(callbacks.Callback):

    """Simple implemnetation cyclic learning rates based on Leslie N. Smith's work
    Cycle learning at from min_lr (note that this will override optimizer.lr)
    to max_lr using custom or predefined function.

    Attributes:
        available_funcs (Dict): list of currently available cycling functions in the class

        decay (float): decay argument to be passed to cycle function

        iteration (int): current epoch - passed to the cycle function

        cycle_function (function | string): If string, must be one of the values in CyclicLR.available_funcs
                                   otherwise this can be a custom cycle function of your own choosing
                                   *NOTE: a custom function must accept the a CyclicLR instance as
                                   an arg.

        max_lr (float): maximum learning rate during cycle
        min_lr (float): minimum learning rate during cycle
        stepsize (int): this represents 1/2 the number of iterations it takes to complete
                         a full cycle (for the learning rate to return to it's original value)
    """
    def triangular(self, decay=1.):
        """with decay == 1. : return learning rate according to triangular policy
        as outlined in Smith's Paper on cyclical learning rates.
        """
        cycle = np.floor(1 + self.iteration  / (2 * self.stepsize))
        x = np.abs(self.iteration  / self.stepsize - 2 * cycle + 1)
        # caculate with decay if any
        return self.min_lr + (self.max_lr - self.min_lr) * np.maximum(0, (1 - x)) * (1 / (decay**(cycle - 1)))

    def triangular2(self):
        return self.triangular(iter_, decay=2)

    available_funcs = {'triangular': triangular,
                       'triangular2': triangular2}

    def calculate_stepsize_range(batch_size, n_samples):
        iter_per_epoch = (n_samples / batch_size)
        return [int((i * iter_per_epoch) // 2) for i in range(2, 10)]

    def __init__(self, stepsize, max_lr=.01, min_lr=.00001, cycle_function='triangular'):
        super(self.__class__, self).__init__()
        self.stepsize = stepsize
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.iteration = 0
        if callable(cycle_function):
            self.func = lambda : cycle_function(self)
        else:
            try:
                self.func = self.available_funcs[cycle_function]
            except KeyError:
                msg = """  could not find the function : {} functions are available. Please use one of those arguments 
                            or provide your own function, which accepts the followin args: epoch_counter,
                            stepsize, min_lr, max_lr, decay""".format(cycle_function, ', '.join(list(self.available_funcs)))
                raise CycleFunctionUnavailable(msg)
        self.hist = defaultdict(list)


    def on_train_begin(self, logs={}):
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def cycle(self):
        new_lr = self.func(self)
        K.set_value(self.model.optimizer.lr, new_lr)
        self.iteration += 1


    def on_batch_end(self, batch, logs={}):
        self.cycle()
        # update batch history
        self.hist['iteration'].append(self.iteration)
        self.hist['lr'].append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.hist[k].append(v)






