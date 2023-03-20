import numpy as np
import dask.array as da
from time import time

lista = np.random.random(1_000_000)
lista2 = np.random.random(1_000_000)

lista_dask = da.from_array(lista)
lista2_dask = da.from_array(lista2)

start_t = time()
print(np.mean(lista + lista2))
print(f'numpy: {time() - start_t}')

start_td = time()
print(da.mean(lista_dask + lista2_dask).compute())
print(f'dask: {time() - start_td}')