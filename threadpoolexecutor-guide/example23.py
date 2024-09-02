# SuperFastPython.com
# set a timeout when getting results from a future
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError

# mock task that will sleep for a moment
def work():
    sleep(1)
    return "all done"

# create a thread pool
with ThreadPoolExecutor() as executor:
    # start one thread
    future = executor.submit(work)
    # get the result from the task, wait for task to complete
    try:
        result = future.result(timeout=0.5)
        print(f'Got Result: {result}')
    except TimeoutError:
        print('Gave up waiting for a result')
