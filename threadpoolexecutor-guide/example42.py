# SuperFastPython.com
# example of recording the execution time of a program
import time

# record the start time
start_time = time.time()
# do work with or without a thread pool
# ....
time.sleep(3)
# record the end time
end_time = time.time()
# report execution time
total_time = end_time - start_time
print(f'Execution time: {total_time:.1f} seconds.')
