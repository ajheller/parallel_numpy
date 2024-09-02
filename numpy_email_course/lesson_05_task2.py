# SuperFastPython.com
# initialize a numpy array in parallel with threads
import concurrent.futures
import time
import numpy
import numba


# fill a portion of a larger array with a value
# @numba.njit(nogil=True)
def fill_subarray(coords, data, value):
    # unpack array indexes
    i1, i2, i3, i4 = coords
    # populate subarray
    data[i1:i2, i3:i4].fill(value)


# record the start time
time_start = time.perf_counter()
# create an empty array
N = 50000
data = numpy.empty((N, N))
# create the thread pool
with concurrent.futures.ThreadPoolExecutor(64) as exe:
    time_start = time.perf_counter()
    # split each dimension (divisor of matrix dimension)
    split = round(N / 11)
    # issue tasks
    for x in range(0, N, split):
        for y in range(0, N, split):
            # determine matrix coordinates
            coords = (x, x + split, y, y + split)
            # issue task
            _ = exe.submit(fill_subarray, coords, data, 1)
# calculate the duration
time_duration = time.perf_counter() - time_start
check = numpy.all(data == 1)
check2 = numpy.sum(data == 1)
# report the duration
print(f"Took {time_duration:.3f} seconds {check} {check2}")
