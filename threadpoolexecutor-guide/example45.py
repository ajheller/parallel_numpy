# SuperFastPython.com
# example of a simple progress indicator
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

# simple progress indicator callback function
def progress_indicator(future):
    print('.', end='', flush=True)

# mock test that works for moment
def task(name):
    sleep(random())

# start the thread pool
with ThreadPoolExecutor(2) as executor:
    # send in the tasks
    futures = [executor.submit(task, i) for i in range(20)]
    # register the progress indicator callback
    for future in futures:
        future.add_done_callback(progress_indicator)
    # wait for all tasks to complete
print('\nDone!')
