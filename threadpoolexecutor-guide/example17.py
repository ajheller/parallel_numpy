# SuperFastPython.com
# configure and report the default number of worker threads
from concurrent.futures import ThreadPoolExecutor
# create a thread pool with a large number of worker threads
pool = ThreadPoolExecutor(500)
# report the number of worker threads
print(pool._max_workers)
