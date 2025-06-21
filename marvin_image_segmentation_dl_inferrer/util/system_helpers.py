#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import os

def get_total_memory_gb():
    '''
    :return: Amount of total memory in GB
    '''
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
    mem_gib = mem_bytes / (1024. ** 3)  # e.g. 3.74
    return mem_gib