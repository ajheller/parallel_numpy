# SuperFastPython.com
# example of submitting follow-up tasks
from time import sleep
from random import random
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

# mock test that works for moment
def task1():
    value = random()
    sleep(value)
    print(f'Task 1: {value}')
    return value

# mock test that works for moment
def task2(value1):
    value2 = random()
    sleep(value2)
    print(f'Task 2: value1={value1}, value2={value2}')
    return value2

# start the thread pool
with ThreadPoolExecutor(5) as executor:
    # send in the first tasks
    futures1 = [executor.submit(task1) for _ in range(10)]
    # process results in the order they are completed
    futures2 = list()
    for future1 in as_completed(futures1):
        # get the result
        result = future1.result()
        # check if we should trigger a follow-up task
        if result > 0.5:
            future2 = executor.submit(task2, result)
            futures2.append(future2)
    # wait for all follow-up tasks to complete
