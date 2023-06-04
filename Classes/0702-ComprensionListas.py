# Comprensión de listas y números random

if __name__ == '__main__':
    lim = 50
    li, ls = 1, 3
    lista = []

    # Obtener números pares de manera incorrecta
    lista = [i for i in range(1, lim+1) if i%2==0]
    print(lista)

    # Obtener números pares de manera correcta
    lista = [i for i in range(2, lim+1, 2)]
    print(lista)

    # Generar números aleatorios con comprensión de listas
    import random

    lista_random = [random.random() for _ in range(3)]
    print(f'Lista con random: {lista_random}')

    lista_rango = [li+random.random()*(ls-li) for _ in range(3)]
    print(f'Lista con un rango calculado:  {lista_rango}')

    lista_matriz = [[random.random() for _ in range(3)] for _ in range(3)]
    print(f'Lista matriz: {lista_matriz}')

    print('\n'.join(','.join(f'{item:.4f}' for item in val) for val in lista_matriz))

    # Ordenar listas
    lista_order = [random.random() for _ in range(5)]
    print(lista_order)

    lista_order.sort()
    print(lista_order)
