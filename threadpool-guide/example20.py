# SuperFastPython.com
# example of calling map without an iterable
from time import sleep
from multiprocessing.pool import ThreadPool
 
# custom function executed in another thread
def task(value):
    # block for a moment
    sleep(1)
    return 'all done'
 
# protect the entry point
if __name__ == '__main__':
    # start the thread pool
    with ThreadPool() as pool:
        # issue all tasks
        for result in pool.map(task):
            print(result)