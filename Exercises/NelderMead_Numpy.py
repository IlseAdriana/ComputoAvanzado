import numpy as np

# Función objetivo
def sphere(simplex):
    # Creación de un arreglo para guardar los fitnesses
    fitnesses = np.zeros(len(simplex), dtype=float)

    # Ciclo para elevar al cuadrado cada valor
    for i, point in enumerate(simplex):
        # Se almacena el fit en el arreglo creado
        fitnesses[i] = np.sum(np.square(point))

    # Unir el arreglo de los fitnesses como la última columna de simplex
    simplex = np.insert(simplex, len(simplex)-1, fitnesses, axis=1)

    # Ordenar el simplex de acuerdo al fit (última columna)
    simplex = simplex[simplex[:, n].argsort()]

    # Retornar valores ordenados
    return simplex[:, :n]
    

# Función que contiene el algoritmo de Nelder-Mead
def algorithm(f, n, alpha=1, gamma=1, rho=0.5, sigma=0.5):
    # SI VAS A UTILIZAR UN VALOR MÁS ADELANTE, GUÁRDALO EN UNA VARIABLE

    # Generar población de n+1 puntos de tamaño n.
    simplex = np.array(np.random.rand(n+1, n)) 

    iterations = 1

    for _ in range(iterations):
        
        # Obtener valores ordenados
        simplex = f(simplex)
        print(simplex)

        # Calcular centros de masa
        m = np.zeros(len(simplex), dtype=float)
        for i, point in enumerate(simplex):
            m[i] = np.sum(point) / n

        print(m)

        # Calcular punto de reflexión contra el peor punto
        r = np.zeros(len(simplex), dtype=float)
        for i in range(n):
            r[i] += m[i] + alpha * (m[i] - simplex[-1])

        print(r)

if __name__ == '__main__':
    f = sphere # Función a evaluar
    n = 2 # Número de dimensiones

    algorithm(f, n)