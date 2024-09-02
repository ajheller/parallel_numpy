# SuperFastPython.com
# example in a callback function for the thread pool
from time import sleep
from multiprocessing.pool import ThreadPool
 
# callback function
def handler(result):
    # report result
    print(f'Got result {result}')
    # fail with an exception
    raise Exception('Something bad happened!')
 
# task executed in a worker thread
def task():
    # block for a moment
    sleep(1)
    # return a value
    return 22
 
# protect the entry point
if __name__ == '__main__':
    # create a thread pool
    with ThreadPool() as pool:
        # issue a task to the thread pool
        result = pool.apply_async(task, callback=handler)
        # wait for the task to finish
        result.wait()