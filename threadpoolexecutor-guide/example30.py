# SuperFastPython.com
# example of handling an exception raised within a task that has a callback
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

# callback function to call when a task is completed
def custom_callback(future):
    print('Custom callback was called')

# mock task that will sleep for a moment
def work():
    sleep(1)
    raise Exception('This is Fake!')
    return "never gets here"

# create a thread pool
with ThreadPoolExecutor() as executor:
    # execute our task
    future = executor.submit(work)
    # add the custom callback
    future.add_done_callback(custom_callback)
    # wait for the task to complete
    wait([future])
    # check the status of the task after it has completed
    running = future.running()
    cancelled = future.cancelled()
    done = future.done()
    print(f'Task running={running}, cancelled={cancelled}, done={done}')
    # get the exception
    exception = future.exception()
    print(f'Exception={exception}')
    # get the result from the task
    try:
        result = future.result()
    except Exception:
        print('Unable to get the result')
