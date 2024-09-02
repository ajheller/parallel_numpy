# SuperFastPython.com
# get the result from a completed future task
from time import sleep
from concurrent.futures import ThreadPoolExecutor

# mock task that will sleep for a moment
def work():
    sleep(1)
    return "all done"

# create a thread pool
with ThreadPoolExecutor() as executor:
    # start one thread
    future = executor.submit(work)
    # get the result from the task, wait for task to complete
    result = future.result()
    print(f'Got Result: {result}')
