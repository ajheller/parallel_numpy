# SuperFastPython.com
# add callbacks to a future, one of which raises an exception
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

# callback function to call when a task is completed
def custom_callback1(future):
    raise Exception('Something bad happened!')
    # never gets here
    print('Callback 1 called.')

# callback function to call when a task is completed
def custom_callback2(future):
    print('Callback 2 called.')

# mock task that will sleep for a moment
def work():
    sleep(1)
    return 'Task is done'

# create a thread pool
with ThreadPoolExecutor() as executor:
    # execute the task
    future = executor.submit(work)
    # add the custom callbacks
    future.add_done_callback(custom_callback1)
    future.add_done_callback(custom_callback2)
    # wait for the task to complete and get the result
    result = future.result()
    # wait for callbacks to finish
    sleep(0.1)
    print(result)
