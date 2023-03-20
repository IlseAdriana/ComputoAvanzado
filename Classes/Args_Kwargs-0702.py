# Funciones

# Funciones con parámetros
def suma(num1, num2):
    return num1 + num2

def resta(num1, num2):
    return num1 - num2

# Funciones con parámetros opcionales (deben tener algo por defecto)
def operacion(num1, num2, op=suma):
    return op(num1, num2)

print(operacion(1.5, 3))
print(operacion(1.5, 3, resta))

# Funciones que reciben *args (tuplas)
def sumar_valores(*args):
    for val in args:
        if isinstance(val, str):
            print('No se permiten cadenas')
            return None
        if val is None:
            print("No se aceptan valores nulos")
            return None
        return sum(args)

print(sumar_valores(1,2,3,4,5,6,7))
print(sumar_valores(1,2,3,4,5,6,7, None))
print(sumar_valores(1,2,3,4,5,6,7, "1"))

# Funciones que reciben **kwargs (diccionarios)
def saludar(**kwargs):
    if 'name' in kwargs:
        return f'Hola {kwargs["name"]}'

print(saludar(name="Ilse"))

