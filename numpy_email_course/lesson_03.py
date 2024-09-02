# SuperFastPython.com
# example of multithreaded matrix multiplication
import os

os.environ["OMP_NUM_THREADS"] = "12"
import time
import numpy

# record the start time
start = time.perf_counter()
# create an array of random values
data1 = numpy.random.rand(8000, 8000)
data2 = numpy.random.rand(8000, 8000)
# matrix multiplication
result = data1.dot(data2)
# calculate and report duration
duration = time.perf_counter() - start
print(f"Took {duration:.3f} seconds")
