# SuperFastPython.com
# example of the submit and use a callback pattern for the ThreadPoolExecutor
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor

# custom task that will sleep for a variable amount of time
def task(name):
    # sleep for less than a second
    sleep(random())
    return name

# custom callback function called on tasks when they complete
def custom_callback(fut):
    # retrieve the result
    print(fut.result())

# start the thread pool
with ThreadPoolExecutor(10) as executor:
    # submit tasks and collect futures
    futures = [executor.submit(task, i) for i in range(10)]
    # register the callback on all tasks
    for future in futures:
        future.add_done_callback(custom_callback)
    # wait for tasks to complete...
