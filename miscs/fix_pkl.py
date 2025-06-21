import fnmatch
from os import listdir
from os.path import isfile, join
import os
import pickle

def recursive_glob(rootdir='.', pattern='*'):
    """Search recursively for files matching a specified pattern.

    Adapted from http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches


root = '/mnt/ebs_xizhn2/Data/Segmentation'

matches = recursive_glob(root, pattern='*.pkl')

count = 0
for match in matches:
    print('{}, count: {}'.format(match, count))

    data = pickle.load(open(match, "rb"))

    newdata = {}
    newdata['BSx256x32x40'] = data['BSx256x32x40']
    newdata['BSx64x128x160'] = data['BSx64x128x160']
    newdata['BSx1x512x640'] = data['BSx32x512x640']

    pickle.dump(newdata, open(match, 'wb'))

    count = count + 1
    zx = 0