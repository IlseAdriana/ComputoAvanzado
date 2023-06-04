import pickle

lista = [1,2,3,4,5, ['Juan'], 'Pedro']

with open('lista.dat', 'wb') as file:
    pickle.dump(lista, file)

with open('lista.dat', 'rb') as file:
    lista2 = pickle.load(file)
    print(lista2)