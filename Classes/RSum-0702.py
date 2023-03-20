# Suma de Rienmann del profesor
if '__name__' == '__main__':
    equation = '2*x+1'
    li, ls = 1, 3
    n = 1

    delta_x = (ls - li) / n

    area = 0
    for i in range(n):
        x = li + i * delta_x
        area += eval(equation) * delta_x

    print(f'Área bajo la curva (normal): {area}')

    ## Suma de Rienmann con comprensión de listas
    print(f'Área bajo la curva (listas): {sum([eval(equation, {"x": li+i*delta_x}) * delta_x for i in range(n)])}')
    
    