from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()    # numero del procesos en los que se esta ejecutando
size = comm.Get_size()    # total de procesos

arreglo=np.asarray([1,2,3], dtype=np.float32)
if rank==0:
  print(rank, arreglo)
  for i in range(size-1):
    comm.Send(arreglo, dest=i+1)
else:
  valor = np.zeros(3, dtype=np.float32)
  comm.Recv([valor, MPI.FLOAT], source=0)
  valor *= 2
  print(rank, np.sum(valor))

if rank!=0:
  comm.Send(valor, dest=0)
else: 
  valor = np.zeros((size-1, 3), dtype=np.float32)
  for i in range(1, size):
    comm.Recv([valor[i-1], MPI.FLOAT], source=i)
  
  print(rank, valor)