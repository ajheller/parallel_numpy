# SuperFastPython.com
# example of the submit and use sequentially pattern for the ThreadPoolExecutor
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor

# custom task that will sleep for a variable amount of time
def task(name):
    # sleep for less than a second
    sleep(random())
    return name

# start the thread pool
with ThreadPoolExecutor(10) as executor:
    # submit tasks and collect futures
    futures = [executor.submit(task, i) for i in range(10)]
    # process task results in the order they were submitted
    for future in futures:
        # retrieve the result
        print(future.result())
