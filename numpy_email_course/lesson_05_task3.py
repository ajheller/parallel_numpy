#! /usr/bin/env python3
#
# initialize a numpy array in parallel with threads
#   inspired by https://superfastpython.com/concurrent-numpy-7-day-course/
#
#   Aaron Heller <aaron.heller@sri.com>
#   2 September 2024
#
# references:
#   https://docs.python.org/3/library/concurrent.futures.html
#   https://superfastpython.com/concurrent-numpy-7-day-course/

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=import-error


import concurrent.futures
import time

import numpy

N = 1 << 16
print(f"{N=}")

VALUE = numpy.pi
DTYPE = numpy.float64


def test0(value, dtype=DTYPE):
    data = numpy.zeros((N, N), dtype=dtype)

    time_start = time.perf_counter()
    data.fill(value)
    time_duration = time.perf_counter() - time_start

    check = numpy.sum(data != value)
    print(f"test0: {time_duration=:.3f} {check=}")

    return time_duration, check


# @numba.njit(nogil=True, boundscheck=False)
def task(data, x, x1, y, y1, value):
    data[x:x1, y:y1].fill(value)


def test1(value, split, pool_size):
    data = numpy.empty((N, N), dtype=DTYPE)
    data.fill(value - 1)  # initialize array with something other than value

    # create the thread pool
    with concurrent.futures.ThreadPoolExecutor(pool_size) as exe:
        time_start = time.perf_counter()
        # issue tasks
        for x in range(0, N, split):
            for y in range(0, N, split):
                _ = exe.submit(task, data, x, x + split, y, y + split, value)
    # calculate the duration
    time_duration = time.perf_counter() - time_start

    check = numpy.sum(data != value)
    # report the duration
    print(f"test1: {time_duration=:.3f} {check=}")
    return time_duration, check


def test2(value, split, pool_size):
    data = numpy.empty((N, N), dtype=DTYPE)
    data.fill(value - 1)  # initialize array with something other than value

    # create the thread pool
    with concurrent.futures.ThreadPoolExecutor(pool_size) as exe:
        time_start = time.perf_counter()
        fs = [
            exe.submit(d.fill, value)
            for d in numpy.reshape(
                data,
                (-1, split, split),
                copy=False,  # raise an exception on copy
            )
        ]
        # fs = [exe.submit(d.fill, value) for d in data.reshape((-1, split * split))]
        concurrent.futures.wait(fs)
        time_duration = time.perf_counter() - time_start

    check = numpy.sum(data != value)
    print(f"test1: {time_duration=:.3f} {check=}")
    return time_duration, check


t0, c0 = test0(VALUE)

t1, c1 = test1(VALUE, round(N / 4), 8)

t2, c2 = test2(VALUE, round(N / 16), 64)

print(f"speedup = {t0 / t1:.3f}")
