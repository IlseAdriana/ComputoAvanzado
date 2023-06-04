from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Número de proceso en el que se está ejecutando
size = comm.Get_size() # Total de procesos

print(f'{rank}/{size}')
print(f"I'm rank {rank} among {size} processes\n")