# SuperFastPython.com
# example of calling submit with a function call
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor

# custom task that will sleep for a variable amount of time
def task():
    # sleep for less than a second
    sleep(random())
    return 'all done'

# start the thread pool
with ThreadPoolExecutor() as executor:
    # submit the task
    future = executor.submit(task())
    # get the result
    result = future.result()
    print(result)