# SuperFastPython.com
# example of issuing tasks to a pool that is closed
from time import sleep
from multiprocessing.pool import ThreadPool
 
# custom function executed in another thread
def task():
    # block for a moment
    sleep(1)
    return 'all done'
 
# protect the entry point
if __name__ == '__main__':
    # start the thread pool
    with ThreadPool() as pool:
        # issue the task
        result = pool.apply_async(task)
        # close the pool
        pool.close()
        # wait for all tasks to finish
        pool.join()
        # issue another task
        result = pool.apply_async(task)