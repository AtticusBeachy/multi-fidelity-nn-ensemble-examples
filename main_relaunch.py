import os
import time

START_TIME = time.perf_counter()

exit_code = 1
ii = 1
MAX_RUNS = 100

while exit_code and ii<=MAX_RUNS:
    print(f"Step {ii} of maximum {MAX_RUNS}")
    exit_code = os.system("python main_e2nn_adaptive_sampling.py")
    print("Exit code: ", exit_code)
    ii+=1
    time.sleep(1)

END_TIME = time.perf_counter()
print("Running time: ", END_TIME - START_TIME)
print(f"Ended on run number {ii-1} ({ii-1} crashes)")
