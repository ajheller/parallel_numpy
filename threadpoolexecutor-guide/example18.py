# SuperFastPython.com
# report the default name of threads in the thread pool
import threading
from concurrent.futures import ThreadPoolExecutor

# a mock task that does nothing
def task(name):
    pass

# create a thread pool
executor = ThreadPoolExecutor()
# execute asks
executor.map(task, range(10))
# report all thread names
for thread in threading.enumerate():
    print(thread.name)
# shutdown the thread pool
executor.shutdown()
