# SuperFastPython.com
# example of handling an exception raised within a task
from time import sleep
from concurrent.futures import ThreadPoolExecutor

# mock task that will sleep for a moment
def work():
    sleep(1)
    raise Exception('Something bad happened!')
    return "never gets here"

# create a thread pool
with ThreadPoolExecutor() as executor:
    # execute our task
    future = executor.submit(work)
    # get the result from the task
    try:
        result = future.result()
    except Exception:
        print('Unable to get the result')
