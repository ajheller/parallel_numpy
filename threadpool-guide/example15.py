# SuperFastPython.com
# exception in one of many tasks issued to the thread pool asynchronously
from time import sleep
from multiprocessing.pool import ThreadPool
 
# task executed in a worker thread
def task(value):
    # block for a moment
    sleep(1)
    # check for failure case
    if value == 2:
        raise Exception('Something bad happened!')
    # report a value
    return value
 
# protect the entry point
if __name__ == '__main__':
    # create a thread pool
    with ThreadPool() as pool:
        # issues tasks to the thread pool
        result = pool.map_async(task, range(5))
        # iterate over the results
        for value in result.get():
            print(value)