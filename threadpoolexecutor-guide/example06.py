# SuperFastPython.com
# example of calling map with two iterables
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor

# custom task that will sleep for a variable amount of time
def task(value1, value2):
    # sleep for less than a second
    sleep(random())
    return (value1, value2)

# start the thread pool
with ThreadPoolExecutor() as executor:
    # submit all tasks
    for result in executor.map(task, ['1', '2', '3'], ['a', 'b', 'c']):
        print(result)
