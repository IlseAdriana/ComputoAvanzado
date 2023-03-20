from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from time import time


def calculate_PI(li, ls):
    val_pi = 0

    for k in range(li, ls+1):
        k_8 = 8 * k
        val_pi += (1 / 16**k) * ( (4 / (k_8+1)) - (2 / (k_8+4)) - (1 / (k_8+5)) - (1 / (k_8+6)) )

    return val_pi

def main():
    it = 50_000 # Iterations number

    # Secuential
    st_time_sec = time()
    print(f'Secuential PI: {calculate_PI(0, it)}')
    print(f'Sec_Time: {time() - st_time_sec}')
    print('\n')

    
    # Threads
    st_time_thr = time()
    n_ops = int(it / cpu_count()) # Operations to be executed by each thread
    results = [] # List to store each thread execution result

    with ThreadPoolExecutor() as exe:
        for i in range(cpu_count()):
            results.append(exe.submit(calculate_PI, i*n_ops, (i+1)*n_ops))

    # Loop to sum the results obtained in each thread execution
    pi_threads = 0
    for value in results:
        pi_threads += value.result()

    print(f'Threads PI: {pi_threads}')
    print(f'Thr_Time: {time() - st_time_thr}')
    print('\n')


    # Processes
    st_time_pro = time()
    pool = Pool()
    results = [] # List to store each process result
    for i in range(cpu_count()):
        results.append(pool.apply_async(calculate_PI, args=(i*n_ops, (i+1)*n_ops)))

    pool.close()
    pool.join()

    pi_processes = 0
    for value in results:
        pi_processes += value.get()
        
    print(f'Processes PI: {pi_processes}')
    print(f'Proc_Time: {time() - st_time_pro}')

    
if __name__ == '__main__':
    main()