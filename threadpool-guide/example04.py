# SuperFastPython.com
# example of the map an iterate results usage pattern
from time import sleep
from random import random
from multiprocessing.pool import ThreadPool
 
# task to execute in a new thread
def task(value):
    # generate a random value
    random_value = random()
    # block for moment
    sleep(random_value)
    # return a value
    return (value, random_value)
 
# protect the entry point
if __name__ == '__main__':
    # create the thread pool
    with ThreadPool() as pool:
        # issue tasks and thread results
        for result in pool.map(task, range(10)):
            print(f'>got {result}')