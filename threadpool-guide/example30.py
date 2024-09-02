# SuperFastPython.com
# example of showing progress in the thread pool with separate tasks
from time import sleep
from random import random
from multiprocessing.pool import ThreadPool
 
# progress indicator for tasks in the thread pool
def progress(results):
    print('.', end='')
 
# task executed in a worker thread
def task():
    # generate a random value
    value = random()
    # block for a moment
    sleep(value)
 
# protect the entry point
if __name__ == '__main__':
    # create and configure the thread pool
    with ThreadPool() as pool:
        # issue many tasks asynchronously to the thread pool
        results = [pool.apply_async(task, callback=progress) for _ in range(20)]
        # close the pool
        pool.close()
        # wait for all issued tasks to complete
        pool.join()
    # report all done
    print('\nDone!')