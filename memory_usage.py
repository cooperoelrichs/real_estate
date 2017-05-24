import sys
sys.path.insert(0, "../")

import os
import psutil


class MU():
    GB_FACTOR = 2 ** (3 * 10)

    def memory_usage():
        pid = os.getpid()
        mem = psutil.Process(pid).memory_info().rss
        return mem, pid

    def gb_pid():
        mem, pid = MU.memory_usage()
        return mem / MU.GB_FACTOR, pid


    def print_memory_usage():
        mem, pid = MU.memory_usage()
        print('%.4f GB of memory used by pid %i.' % (gbs, pid))
