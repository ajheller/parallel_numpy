# SuperFastPython.com
# example of safely stopping all tasks in the thread pool
from time import sleep
from threading import Event
from multiprocessing.pool import ThreadPool
 
# task executed in a worker thread
def task(identifier, event):
    print(f'Task {identifier} running')
    # run for a long time
    for i in range(10):
        # block for a moment
        sleep(1)
        # check if the task should stop
        if event.is_set():
            print(f'Task {identifier} stopping...')
            # stop the task
            break
    # report all done
    print(f'Task {identifier} Done')
 
# protect the entry point
if __name__ == '__main__':
    # create the shared event
    event = Event()
    # create and configure the thread pool
    with ThreadPool() as pool:
        # prepare arguments
        items = [(i,event) for i in range(4)]
        # issue tasks asynchronously
        result = pool.starmap_async(task, items)
        # wait a moment
        sleep(2)
        # safely stop the issued tasks
        print('Safely stopping all tasks')
        event.set()
        # wait for all tasks to stop
        result.wait()
        print('All stopped')