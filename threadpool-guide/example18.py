# SuperFastPython.com
# example of calling submit with a function call
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
        result = pool.apply_async(task())
        # get the result
        value = result.get()
        print(value)