# SuperFastPython.com
# example of an exception in a thread pool initializer function
from time import sleep
from random import random
from threading import current_thread
from concurrent.futures import ThreadPoolExecutor

# function for initializing the worker thread
def initializer_worker():
    # raise an exception
    raise Exception('Something bad happened!')

# a mock task that sleeps for a random amount of time less than one second
def task(identifier):
    sleep(random())
    # get the unique name
    return identifier

# create a thread pool
with ThreadPoolExecutor(max_workers=2, initializer=initializer_worker) as executor:
    # execute tasks
    for result in executor.map(task, range(10)):
        print(result)
