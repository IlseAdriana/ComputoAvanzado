import threading
import multiprocessing.pool as mp

def saludar(nombre):
    print(f'Hola {nombre}')

if __name__ == '__main__':
    nombres = ['Pedro', 'Juan, María']

    for i in range(10):
        nombres.append(threading.Thread(target=saludar, args=(i, )))

    for hilo in nombres:
        hilo.join()

    print('Se terminó el programa')