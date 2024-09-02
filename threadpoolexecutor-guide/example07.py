# SuperFastPython.com
# example of calling map and not iterating the results
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor

# custom task that will sleep for a variable amount of time
def task(value):
    # sleep for less than a second
    sleep(random())
    print(f'Done: {value}')
    return value

# start the thread pool
with ThreadPoolExecutor() as executor:
    # submit all tasks
    executor.map(task, range(5))
print('All done!')
