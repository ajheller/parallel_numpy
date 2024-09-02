# SuperFastPython.com
# example of thread local storage for worker threads
from time import sleep
from random import random
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

# function for initializing the worker thread
def initializer_worker(local):
    # generate a unique value for the worker thread
    local.key = random()
    # store the unique worker key in a thread local variable
    print(f'Initializing worker thread {local.key}')

# a mock task that sleeps for a random amount of time less than one second
def task(local):
    # access the unique key for the worker thread
    mykey = local.key
    # make use of it
    sleep(mykey)
    return f'Worker using {mykey}'

# get the local context
local = threading.local()
# create a thread pool
executor = ThreadPoolExecutor(max_workers=2, initializer=initializer_worker, initargs=(local,))
# dispatch asks
futures = [executor.submit(task, local) for _ in range(10)]
# wait for all threads to complete
for future in futures:
    result = future.result()
    print(result)
# shutdown the thread pool
executor.shutdown()
print('done')
