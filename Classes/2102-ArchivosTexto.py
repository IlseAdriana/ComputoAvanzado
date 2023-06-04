"""
Archivos
-- Texto (Aquellos que son legibles en cualquier editor de texto)
-- Binarios (Aquellos que no son legibles para el ser humano
                y se necesita mayormente un programa de terceros)

path (dirección o ubicación) nombre extensión
"""

file = open('info.txt', 'w')
file.write('Hola Mundo')
file.write('\n')
file.close()

file = open('info.txt', 'r')
print(file.readlines())
for line in file.readlines():
    print(line)

# Cerrado automático
with open('info.txt', 'a') as file:
    file.write('Hola Mundo 2\n')

with open('info.txt', 'r') as file:
    for line in file.readlines():
        print(line.replace('\n', ''))