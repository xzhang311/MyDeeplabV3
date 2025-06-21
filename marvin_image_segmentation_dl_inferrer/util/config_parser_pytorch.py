#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import json
import os
import errno

class ConfigParserPyTorch:
    def __init__(self, path):
        self.path = path
        self.required_params = ('arch', 'batch_size', 'num_workers', 'image_width', 'image_height', 'gpu_support', 'dataset')
        self.params = self.load_params()

    def load_params(self):
        print('Loading config file at: '.format(self.path))
        if not os.path.isfile(str(self.path)):
            raise IOError('Invalid config file path')
        with open(self.path, 'r') as f:
            params = json.loads(f.read())
        return params

    def validate_params(self):
        '''
        Validate all params needed in training process are available. The exact checking process should be specific to
        each training process.
        :return:
        '''
        for param in self.required_params:
            if param not in self.params:
                print(param + ' ' + 'is not provided in parameter file')
                return False
            else:
                return True
