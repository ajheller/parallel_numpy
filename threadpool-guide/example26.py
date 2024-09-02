# SuperFastPython.com
# example of waiting for all tasks in a batch to finish
from time import sleep
from multiprocessing.pool import ThreadPool
 
# task executed in a worker thread
def task(identifier):
    # block for a moment
    sleep(0.5)
    # report done
    print(f'Task {identifier} done')
 
# protect the entry point
if __name__ == '__main__':
    # create and configure the thread pool
    with ThreadPool() as pool:
        # issue tasks into the thread pool
        result = pool.map_async(task, range(10))
        # wait for tasks to complete
        result.wait()
        # report all tasks done
        print('All tasks are done')
    # thread pool is closed automatically