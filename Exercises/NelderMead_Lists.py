from random import random

# Función objetivo
def sphere(simplex): 
    return sum([value**2 for value in simplex])

# Función para calcular los centros de masa
def mass_centers(simplex):
    return [sum(simplex[i]) / n for i in range(n)]

# Función para calcular el punto de reflexión
def reflection(simplex, m, alpha):
    return [m[i] + alpha * (m[i] - simplex[n][i]) for i in range(n)]

# Función para calcular el punto de expansión
def expansion(r, m, gamma):
    return [r[i] + gamma * (r[i] - m[i]) for i in range(n)]

# Función para calcular el punto de contracción
def contraction(r, m, rho):
    return [rho * r[i] + (1 - rho) * m[i] for i in range(n)]

# Función para realizar el encogimiento
def shrink(simplex, sigma):
    return [simplex[0] + sigma * (simplex[i] - simplex[0]) for i in range(1, n)]

# Función que contiene el algoritmo de Nelder-Mead
def algorithm(f, n, li, ls, alpha=1, gamma=1, rho=0.5, sigma=0.5):
    # Crear población de n+1 puntos de tamaño n.
    simplex = [[li + random() * (ls - li) for _ in range(n)] for _ in range(n + 1)]
    print(simplex)

    iterations = 100

    for _ in range(iterations):
        # SI VAS A UTILIZAR UN VALOR MÁS ADELANTE, GUÁRDALO EN UNA VARIABLE

        # Ordenar  elementos conforme al resutado de la evaluación en la función.
        simplex.sort(key = f)

        # Calcular el centro de masa de los n mejores puntos.
        m = mass_centers(simplex)

        # Calcular el punto de reflexión
        r = reflection(simplex, m, alpha)

        # Si la evaluación del punto de reflexión es mejor que la del mejor punto y la del peor
        if f(simplex[0]) < f(r) < f(simplex[n]):
            # Reemplazar el peor punto por r, haciendo la reflexión.
            simplex[n] = r
        
        # Sino, y la evaluación del punto de reflexión es mejor o igual a la del mejor punto:
        elif f(r) <= f(simplex[0]):
            # Calcular el punto de expansión.
            e = expansion(r, m, gamma)
            # Si la evaluación del punto de expansión es mejor a la de reflexión:
            if (f(e) < f(r)):
                # Reemplazar el peor punto por e, haciendo la expansión.
                simplex[n] = e
                print("Expansion")
            else:
                # Reemplazar el peor punto por r, haciendo la reflexión.
                simplex[n] = r
                print("Reflexión")

        # Sino, fue peor que el mejor punto y que el peor punto,
        # entonces se calcula el punto de la contracción del simplex.
        elif f(r) >= f(simplex[n-1]):
            b = True
            c = contraction(r, m, rho)
            # Si el punto de contracción es menor al punto de reflexión
            if f(c) < f(r):
                # Reemplazamos el peor punto por el punto de contracción
                simplex[n] = c
                b = False
                print("Contracción")
        # Sino, entonces se encoge el simplex hacia el mejor punto
        elif b == True:
            simplex = shrink(simplex, sigma)
            print("Encogimiento")

    return simplex[0]

if __name__ == '__main__':
    f = sphere # Función a evaluar
    n = 3 # Número de dimensiones

    optimun = algorithm(f, n, -10, 10)
    print(f'Punto óptimo: {optimun}')