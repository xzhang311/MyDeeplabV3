#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import json
import os
import errno

class JsonLoaderPyTorch:
    def __init__(self, path):
        self.path = path
        self.required_params = (
        'arch', 'epochs', 'learning_rate', 'decay', 'optimizer', 'batch_size', 'num_workers', 'dataset_root',
        'pretrained', \
        'image_width', 'image_height', 'gpu_support')

    def load_params(self):
        if not os.path.isfile(self.path):
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

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