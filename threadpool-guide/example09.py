# SuperFastPython.com
# example of an exception raised in the worker initializer function
from time import sleep
from multiprocessing.pool import ThreadPool
 
# function for initializing the worker thread
def init():
    # raise an exception
    raise Exception('Something bad happened!')
 
# task executed in a worker thread
def task():
    # block for a moment
    sleep(1)
 
# protect the entry point
if __name__ == '__main__':
    # create a thread pool
    with ThreadPool(initializer=init) as pool:
        # issue a task
        pool.apply(task)