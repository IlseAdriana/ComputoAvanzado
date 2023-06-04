from pandas import read_csv
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import multiprocessing as mp

def normalize(data):
    for x in data:
        x /= np.linalg.norm(x)

    return data


def eval_indivPerceptron(z, weights):
    f_z = []
    for w in weights:
        f_z.append(1 if z @ w >= 0 else -1)

    return f_z


def update_weights(z, y, y_pred, weights, eta, epsilon=0.1, gamma=0.05, mu=1.0):
    # Regla P-Delta
    for w in weights:
        dot_prod = w @ z # Producto-punto del patron actual y los pesos del perceptron w
        if y_pred > (y + epsilon) and dot_prod >= 0:
            eta *= -z
        elif y_pred < (y - epsilon) and dot_prod < 0:
            eta *= z
        elif y_pred <= (y + epsilon) and 0 < dot_prod < gamma:
            eta *= mu * z
        elif y_pred >= (y - epsilon) and -gamma < dot_prod < 0:
            eta *= mu * -z
        else:
            eta *= 0
        w += eta # Actualición del peso
        w = normalize(w) # Normalizar vector

    return weights


def parallel_perceptron(n_perceptrons, patterns, labels, epochs):
    # Crear vectores iniciales de pesos
    weights = np.asarray([np.random.uniform(-0.5, 0.5, patterns.shape[1]) for _ in range(n_perceptrons)])

    for t in range(1, epochs+1):
        acc = 0 # Almacenar aciertos

        eta = 1 / (4 * np.sqrt(t))

        for z, y in zip(patterns, labels):
            f_z = eval_indivPerceptron(z, weights)

            p = np.sum(f_z)

            y_pred = 1 if p >= 0 else -1

            if y_pred == y:
                acc += 1
            else:
                weights = update_weights(z, y, y_pred, weights, eta)

        print(f'Epoch {t} | Correct {acc}/{len(patterns)} | Accuracy {acc/len(patterns)*100:.2f}%')


def main():
    # Iris-plant biclass dataset
    df = read_csv('Projects/datasets/iris.data.biclass.csv', sep=',')
    X = np.asarray(df.iloc[:,:4])
    y = np.where(df.iloc[:, -1] == 'Iris-setosa', 0, 1)
    
    # Make-moons dataset
    # X, y = make_moons(n_samples=100, noise=0.2, shuffle=True)

    X = normalize(X) # Normalize data
    n_proc = mp.cpu_count() - 1 # Cantidad de procesadores (cada uno representa un perceptron)

    # Crear conjuntos de entrenamiento y prueba
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y)

    epochs = 10

    # Training
    print('Training')
    parallel_perceptron(n_proc, X_tr, y_tr, epochs)

    # Testing
    print('\nTesting')
    parallel_perceptron(n_proc, X_te, y_te, epochs)


if __name__ == '__main__':
    main()