# SuperFastPython.com
# example of the submit and wait for all with shutdown pattern for the ThreadPoolExecutor
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor

# custom task that will sleep for a variable amount of time
def task(name):
    # sleep for less than a second
    sleep(random())
    # display the result
    print(name)

# start the thread pool
executor = ThreadPoolExecutor(10)
# submit tasks and collect futures
futures = [executor.submit(task, i) for i in range(10)]
# wait for all tasks to complete
executor.shutdown()
print('All tasks are done!')
