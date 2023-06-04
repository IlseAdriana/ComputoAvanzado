import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() #num del procesos en los que sde está ejecutando
size = comm.Get_size() #total de procesos

#arreglo = np.random.randint(1,10,20)
#print(arreglo)

if rank == 0:
    arreglo = np.asarray([1,2,3], dtype=np.float32)

    for i in range(1, size):
        comm.Send(arreglo, dest=1)
elif rank == 1:
    info = np.zeros(3, dtype=np.float32)
    comm.Recv([info, MPI.FLOAT], source=0)
    print(info)
else:
    print(f'{rank} no tiene nada')