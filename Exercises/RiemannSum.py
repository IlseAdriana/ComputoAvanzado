# ¿Qué recibe?: Función a evaluar, LimInf, LimSup, NumRectángulos

if __name__ == '__main__':
    equation = 'x**2' # Función a evaluar
    li = 1 # Límite Inferior
    ls = 100 # Límite Superior
    n = 20 # Número de rectángulos

    # Ancho del rectángulo
    delta_x = (ls - li) / n

    x = 0
    area = 0
    for i in range(n):
        # Valor del extremo derecho del rectángulo
        x += delta_x

        # Evaluar x en f(x)
        area += eval(equation)

    # Multiplicar por delta_x al terminar la sumatoria
    area *= delta_x

    print(f'Área bajo la curva: {area}')