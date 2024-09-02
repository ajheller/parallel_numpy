# SuperFastPython.com
# example of checking for an exception raised in the task
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
        # issue a task
        result = pool.apply_async(task)
        # wait for the task to finish
        result.wait()
        # check for a failure
        if result.successful():
            # get the result
            value = result.get()
            # report the result
            print(value)
        else:
            # report the failure case
            print('Unable to get the result')