# SuperFastPython.com
# example of handling an exception raise within a task
from time import sleep
from concurrent.futures import ThreadPoolExecutor

# mock task that will sleep for a moment
def work():
    sleep(1)
    try:
        raise Exception('Something bad happened!')
    except Exception:
        return 'Unable to get the result'
    return "never gets here"

# create a thread pool
with ThreadPoolExecutor() as executor:
    # execute our task
    future = executor.submit(work)
    # get the result from the task
    result = future.result()
    print(result)
