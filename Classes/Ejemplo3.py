
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() #num del procesos en los que sde está ejecutando
size = comm.Get_size() #total de procesos

#arreglo = np.random.randint(1,10,20)
#print(arreglo)

arreglo = np.zeros(10, dtype= np.int32)

if rank == 0:
    arreglo = np.random.randint(0,10, size=10,  dtype=np.int32)

comm.Bcast(arreglo, 0) #broadcast

print(arreglo)