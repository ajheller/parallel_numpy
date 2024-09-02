# SuperFastPython.com
# set a custom thread name prefix for all threads in the pool
import threading
from concurrent.futures import ThreadPoolExecutor

# a mock task that does nothing
def task(name):
    pass

# create a thread pool with a custom name prefix
executor = ThreadPoolExecutor(thread_name_prefix='TaskPool')
# execute asks
executor.map(task, range(10))
# report all thread names
for thread in threading.enumerate():
    print(thread.name)
# shutdown the thread pool
executor.shutdown()
