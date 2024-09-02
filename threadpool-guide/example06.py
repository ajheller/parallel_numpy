# SuperFastPython.com
# example of the map_async and forget usage pattern
from time import sleep
from random import random
from multiprocessing.pool import ThreadPool
 
# task to execute in a new thread
def task(value):
    # generate a random value
    random_value = random()
    # block for moment
    sleep(random_value)
    # prepare result
    result = (value, random_value)
    # report results
    print(f'>task got {result}')
 
# protect the entry point
if __name__ == '__main__':
    # create the thread pool
    with ThreadPool() as pool:
        # issue tasks to the thread pool
        _ = pool.map_async(task, range(10))
        # close the pool
        pool.close()
        # wait for all tasks to complete
        pool.join()