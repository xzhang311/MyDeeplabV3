#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import datetime
import os

class NNTrainer(object):
    '''
    Class that defines skeleton of trainer
    '''
    def __init__(self):
        '''
        :param params: Params used in training process. Params should be saved in a dict format.
        '''

    def run(self):
        '''
        Running training process. The actual behavior of the training process should be defined by
        subclasses.
        :return:
        '''
        raise NotImplementedError