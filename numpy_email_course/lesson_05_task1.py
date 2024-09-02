# SuperFastPython.com
# initialize a numpy array slowly

import os
import time

os.environ["OMP_NUM_THREADS"] = "12"  # has no effect
import numpy

data = numpy.empty((50000, 50000))

# record the start time
time_start = time.perf_counter()
# create a new matrix and fill with 1
data.fill(1.0)
# calculate and report duration
time_duration = time.perf_counter() - time_start
# report progress
print(f"Took {time_duration:.3f} seconds")
