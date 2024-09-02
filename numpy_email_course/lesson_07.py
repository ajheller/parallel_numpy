# SuperFastPython.com
# share numpy array between processes via a shared rawarray
import multiprocessing
import numpy


# task executed in a child process
def task(array):
    # create a new numpy array backed by the raw array
    data = numpy.frombuffer(array, dtype=numpy.double, count=len(array))
    # check the contents
    print(f"Child {data[:10]}")
    # increment the data
    data[:] += 1
    # confirm change
    print(f"Child {data[:10]}")


# protect the entry point
if __name__ == "__main__":
    # define the size of the numpy array
    n = 10000000
    # create the shared array
    array = multiprocessing.RawArray("d", n)
    # create a new numpy array backed by the raw array
    data = numpy.frombuffer(array, dtype=numpy.double, count=len(array))
    # populate the array
    data.fill(1.0)
    # confirm contents of the new array
    print(data[:10], len(data))
    # create a child process
    child = multiprocessing.Process(target=task, args=(array,))
    # start the child process
    child.start()
    # wait for the child process to complete
    child.join()
    # check some data in the shared array
    print(data[:10])
