# SuperFastPython.com
# example of asynchronous tasks failing silently in the thread pool
from time import sleep
from multiprocessing.pool import ThreadPool
 
# task executed in a worker thread
def task():
    # block for a moment
    sleep(1)
    # fail
    raise Exception('Something bad happened')
    # report a message
    print(f'Task done')
 
# protect the entry point
if __name__ == '__main__':
    # create and configure the thread pool
    with ThreadPool() as pool:
        # issue an asynchronous task into the thread pool
        result = pool.apply_async(task)
        # wait for all tasks to finish
        result.wait()