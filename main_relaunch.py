import os
import time

START_TIME = time.perf_counter()

exit_code = 1
i = 0
MAX_RUNS = 100

while exit_code and i < MAX_RUNS:
    i+=1
    print(f"Step {i} of maximum {MAX_RUNS}")
    exit_code = os.system("python3 main_e2nn_adaptive_sampling.py")
    print("Exit code: ", exit_code)
    time.sleep(1)

END_TIME = time.perf_counter()
print("Running time: ", END_TIME - START_TIME)
print(f"Ended on run number {i} ({i+bool(exit_code)-1} crashes)")
