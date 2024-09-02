# SuperFastPython.com
# example that combines a thread pool and blas threads
import os

os.environ["OMP_NUM_THREADS"] = "4"
import time
import numpy
import concurrent.futures


# function that defines a single task
def task():
    # simulate loading arrays
    item1 = numpy.random.rand(2000, 2000)
    item2 = numpy.random.rand(2000, 2000)
    # perform multithreaded operation
    result = item1.dot(item2)
    # return the result
    return result


# record the start time
time_start = time.perf_counter()
# create thread pool
with concurrent.futures.ThreadPoolExecutor(2) as exe:
    # issue tasks and gather results
    results = [exe.submit(task) for _ in range(100)]
# calculate the duration
time_duration = time.perf_counter() - time_start
# report the duration
print(f"Took {time_duration:.3f} seconds")
