# SuperFastPython.com
# example of stopping running tasks using an event
from time import sleep
from threading import Event
from concurrent.futures import ThreadPoolExecutor

# mock target task function
def work(event):
    # pretend read data for a long time
    for _ in range(100):
        # pretend to read some data
        sleep(1)
        # check the status of the flag
        if event.is_set():
            # shut down this task now
            print("Not done, asked to stop")
            return
    return "All done!"

# create an event to shut down all running tasks
event = Event()
# create a thread pool
executor = ThreadPoolExecutor(5)
# execute all of our tasks
futures = [executor.submit(work, event) for _ in range(50)]
# wait a moment
print('Tasks are running...')
sleep(2)
# cancel all scheduled tasks
print('Cancelling all scheduled tasks...')
for future in futures:
    future.cancel()
# stop all currently running tasks
print('Trigger all running tasks to stop...')
event.set()
# shutdown the thread pool and wait for all tasks to complete
print('Shutting down...')
executor.shutdown()
