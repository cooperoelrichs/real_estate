import sys
sys.path.insert(0, "../")

import os
import psutil


class MU():
    GB_FACTOR = 2 ** (3 * 10)

    def process_memory_usage():
        pid = MU.get_pid()
        mem = psutil.Process(pid).memory_info().rss
        return mem

    def pmu():
        return MU.to_gb(MU.process_memory_usage())

    def get_pid():
        return os.getpid()

    def memory_available():
        mem = psutil.virtual_memory().available
        return mem

    def ma():
        return MU.to_gb(MU.memory_available())

    def to_gb(mem):
        return mem / MU.GB_FACTOR

    def object_size(x):
        return MU.to_gb(sys.getsizeof(x))

    def df_size(x):
        return MU.to_gb(x.memory_usage(index=True).values.sum())

    def print_memory_usage():
        gb = MU.pmu()
        print(
            '%.4f GB of memory used by pid %i., %.4fGB of memory available' %
            (gb, MU.get_pid()), MU.ma())
