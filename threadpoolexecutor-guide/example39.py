# SuperFastPython.com
# example of calling map without an iterable
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor

# custom task that will sleep for a variable amount of time
def task(value):
    # sleep for less than a second
    sleep(random())
    return value

# start the thread pool
with ThreadPoolExecutor() as executor:
    # submit all tasks
    for result in executor.map(task):
        print(result)
