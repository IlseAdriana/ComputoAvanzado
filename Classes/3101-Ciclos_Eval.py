# Ciclos y evaluación de parámetros

if __name__ == '__main__':
    lista = [1, 2, 3, 4, 5]

    # Forma incorrecta de recorrer elementos
    for i in range(len(lista)):
        print(lista[i], end=", ")

    # Forma correcta de recorrer elementos de una estructura lineal
    print()
    for item in lista:
        print(item, end=", ")

    # Ciclo para recorrer elementos y obtener el índice del actual
    print()
    for i, item in enumerate(lista):
        if i == (len(lista) - 1):
            print(item, end="\n")
        else:
            print(item, end=", ")

    # Captura de datos
    ecuacion = input(("Introduzca una ecuacion (basada en x): "))
    x = float(input("Introduzca el valor de x: "))

    # Evaluación de una función dada con un valor x
    print(eval(ecuacion))
