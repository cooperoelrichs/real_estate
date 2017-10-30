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
        return MU.to_gb(mem), pid

    def to_gb(mem):
        return mem / MU.GB_FACTOR

    def object_size(x):
        return MU.to_gb(sys.getsizeof(x))

    def df_size(x):
        return MU.to_gb(x.memory_usage(index=True).values.sum())

    def print_memory_usage():
        gbs, pid = MU.gb_pid()
        print('%.4f GB of memory used by pid %i.' % (gbs, pid))
