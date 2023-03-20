from multiprocessing import cpu_count
import multiprocessing as mp

valores = []
def calculatePI(a, b, step):
    val_pi = 1
    for i in range(a, b, step):
        n_dos = i * 2
        val_pi *= (n_dos ** 2) / ((n_dos - 1) * (n_dos + 1))
    valores.append((a, val_pi))

    return val_pi
    
if __name__ == '__main__':
    n = 10_000_000
    pi = 2

    pool = mp.Pool()
    vals = []
    for i in range(cpu_count()):
        vals.append(pool.apply_async(calculatePI, args=(i+1, n, cpu_count(),)))
    
    # for i in range(cpu_count()):
        # vals.append(pool.apply(calculatePI, args=(i+1, n, cpu_count(),)))

    pool.close()
    pool.join()

    for val in vals:
        pi *= val.get()
        
    print(pi)
