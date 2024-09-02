# SuperFastPython.com
# example of an error while joining the pool
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
        # wait for all tasks to finish
        pool.join()