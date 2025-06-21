#!/apollo/sbin/envroot "$ENVROOT/bin/python"

class NNInferrer(object):
    '''
    Class that defines skeleton of inferrer
    '''
    def __init__(self):
        '''
        :param params: Params used in inference process. Params should be saved in a dict format.
        '''

    def run(self):
        '''
        Run inference process.
        :return:
        '''
        raise NotImplementedError
