# SuperFastPython.com
# example of shutting down within a context manager
from time import sleep
from concurrent.futures import ThreadPoolExecutor

# mock test that works for moment
def task(name):
    sleep(2)
    print(f'Done: {name}')

# start the thread pool
with ThreadPoolExecutor(1) as executor:
    # send some tasks into the thread pool
    print('Sending in tasks...')
    futures = [executor.submit(task, i) for i in range(10)]
    # explicitly shutdown within the context manager
    print('Shutting down...')
    executor.shutdown(wait=False, cancel_futures=True)
    # shutdown called again here when context manager exited
    print('Waiting...')
print('Doing other things...')
