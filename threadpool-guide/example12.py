# SuperFastPython.com
# example of handling an exception raised within a task in the caller
from time import sleep
from multiprocessing.pool import ThreadPool
 
# task executed in a worker thread
def task():
    # block for a moment
    sleep(1)
    # fail with an exception
    raise Exception('Something bad happened!')
    # unreachable return value
    return 'Never gets here'
 
# protect the entry point
if __name__ == '__main__':
    # create a thread pool
    with ThreadPool() as pool:
        try:
            # issue a task and get the result
            value = pool.apply(task)
            # report the result
            print(value)
        except Exception:
            print('Unable to get the result')