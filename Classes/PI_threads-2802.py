from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor


def calculatePI(a, b):
    val_pi = 1
    for i in range(a, b+1):
        n_dos = i * 2
        val_pi *= (n_dos ** 2) / ((n_dos - 1) * (n_dos + 1))
    
    return val_pi
    
    
if __name__ == '__main__':
    n = 1000
    pi = 2
    # pi *= calculatePI(1, n)

    tam = (int) (n/cpu_count())
    vals = []
    with ThreadPoolExecutor() as ejecutar:
        for i in range(cpu_count()):
            vals.append(ejecutar.submit(calculatePI, 1+i*tam, (i+1)*tam))

    for val in vals:
        pi *= val.result()
        
    print(pi)
