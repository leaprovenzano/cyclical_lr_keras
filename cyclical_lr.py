import numpy as np

from keras import callbacks
from keras import backend as K



class CycleFunctionUnavailable(Exception):
    """Error CyclicLr callback func arg"""
    


class CyclicLR(callbacks.Callback):

    """Simple implemnetation cyclic learning rates based on Leslie N. Smith's work
    Cycle learning at from min_lr (note that this will override optimizer.lr)
    to max_lr using custom or predefined triangular function.

    Attributes:
        available_funcs (Dict): list of currently available cycling functions in the class

        decay (float): decay argument to be passed to cycle function

        epoch_counter (int): current epoch - passed to the cycle function

        cycle_function (function | string): If string, must be one of the values in CyclicLR.available_funcs
                                   otherwise this can be a custom cycle function of your own choosing
                                   *NOTE: a custom function must accept the following arguments :
                                   [epoch_counter, stepsize, min_lr, max_lr, decay]

        max_lr (float): maximum learning rate during cycle
        min_lr (float): minimum learning rate during cycle
        stepsize (int): this represents 1/2 the number of epochs it takes to complete
                         a full cycle (for the learning rate to return to it's original value)
        decay (float): decay factor passed to cycle_function 
    """

    def triangular(epoch_counter, stepsize, min_lr, max_lr, decay=.0):
        """with decay == 0. : return learning rate according to triangular policy
        as outlined in Smith's Paper on cyclical learning rates.

        with decay > 0. : uses a policy similar to triangular2 (again see the text) 
        but decays both max_lr and min_lr as opposed to the distance between the two as outlined
        in the text.
        """
        cycle = np.floor(1 + epoch_counter//(2* stepsize))
        x = np.abs(epoch_counter/stepsize - 2 * cycle +1)
        # caculate with decay if any
        min_lr -= (min_lr * (decay * cycle)) if (min_lr * (decay * cycle)) < min_lr  else 0
        max_lr -= (max_lr * (decay * cycle))
        return (min_lr + (max_lr - min_lr)  * max(0, (1-x)))

    available_funcs = {'triangular' : triangular}


    def __init__(self, stepsize, max_lr=.01, min_lr=.00001, cycle_function='triangular', decay=0.):
        super(CyclicLR, self).__init__()
        self.stepsize = stepsize
        self.max_lr = max_lr
        self.min_lr = min_lr        
        self.epoch_counter= 0
        self.decay = decay
        if callable(cycle_function):
            self.func = cycle_function
        else:
            try:
                self.func = self.available_funcs[cycle_function]
            except KeyError:
                msg =  """  could not find the function : {} functions are available. Please use one of those arguments 
                            or provide your own function, which accepts the followin args: epoch_counter,
                            stepsize, min_lr, max_lr, decay""".format(cycle_function, ', '.join(list(self.available_funcs)))
                raise CycleFunctionUnavailable(msg)

    def cycle(self):                                                                        
        new_lr =  self.func(self.epoch_counter, self.stepsize, self.min_lr, self.max_lr, self.decay)
        K.set_value(self.model.optimizer.lr, new_lr )
        self.epoch_counter +=1
                                                                                    
    def on_epoch_begin(self, batch, logs={}):
        self.cycle()

    def on_epoch_end(self, batch, logs={}):
        #log learning rate in keras.history object
        logs['lr'] = K.get_value(self.model.optimizer.lr)


    
    