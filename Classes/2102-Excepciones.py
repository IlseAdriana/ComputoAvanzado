"""
Excepciones
-- Diseño
----- Léxico
----- Sintaxis
----- Semántica
-- Ejecución
"""

num1 = 10
num2 = 0

try:
    div = num1 / num2
    print(div)
except TypeError as err:
    print('Error de tipo de datos', err)
except ZeroDivisionError as err:
    print('Error de división entre 0', err)
except Exception as err:
    print('Error al efectuar la operación', err)

######################################################

def div(a, b):
    if b==0:
        raise Exception("El valor de b es 0 y no se puede dividir")
    return a/b

try:
    print(div(10, 2))
    print(div(10, 0))
except Exception as err:
    print(err)

num = 0
assert num != 0
print(10/num)