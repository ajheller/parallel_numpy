# SuperFastPython.com
# example of getting the first result from the thread pool with a queue
from random import random
from time import sleep
from queue import SimpleQueue
from multiprocessing.pool import ThreadPool
 
# task executed in a worker thread
def task(identifier):
    # generate a value
    value = 2 + random() * 10
    # report a message
    print(f'Task {identifier} executing with {value}')
    # block for a moment
    sleep(value)
    # return the generated value
    queue.put((identifier, value))
 
# protect the entry point
if __name__ == '__main__':
    # create the shared queue
    queue = SimpleQueue()
    # create and configure the thread pool
    with ThreadPool() as pool:
        # issue many tasks
        _ = pool.map_async(task, range(30))
        # get the first result, blocking
        identifier, value = queue.get()
        # report the first result
        print(f'First result: identifier={identifier}, value={value}')