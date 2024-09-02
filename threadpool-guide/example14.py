# SuperFastPython.com
# exception in one of many tasks issued to the thread pool synchronously
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
        for result in pool.map(task, range(5)):
            print(result)