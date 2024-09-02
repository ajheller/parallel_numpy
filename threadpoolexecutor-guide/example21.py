# SuperFastPython.com
# check the status of a Future object for task executed by a thread pool
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

# mock task that will sleep for a moment
def work():
    sleep(0.5)

# create a thread pool
with ThreadPoolExecutor() as executor:
    # start one thread
    future = executor.submit(work)
    # confirm that the task is running
    running = future.running()
    done = future.done()
    print(f'Future running={running}, done={done}')
    # wait for the task to complete
    wait([future])
    # confirm that the task is done
    running = future.running()
    done = future.done()
    print(f'Future running={running}, done={done}')
