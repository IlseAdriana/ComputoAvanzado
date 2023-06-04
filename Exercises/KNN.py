import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from statistics import mode

def calc_distance(x, y):
    return np.sqrt(np.sum(np.square((x - y))))

def knn(k, X_tr, y_tr, patterns, labels):
    tr_size = len(X_tr) # Train-set size
    pa_size = len(patterns) # Patterns size

    # Array to store the acurracy per class
    accuracy = np.zeros(len(np.unique(y_tr)), dtype=int)

    for i in range(pa_size):
        # Array to store distances and the label of the pattern
        distances = np.zeros((tr_size, 2))

        # Distance of pattern_i against each X_tr
        for j in range(tr_size):
            distances[j, 0] = calc_distance(patterns[i], X_tr[j])
            distances[j, 1] = y_tr[j]

        # Sort distances
        distances = distances[distances[:, 0].argsort()]
        
        # Choose k-first distances with the best value
        k_neighbors = distances[:k]

        # Get the label
        predicted_label = int(mode(k_neighbors[:, 1]))

        # See if the predicted label was corrected classified
        if (predicted_label == labels[i]):
            accuracy[predicted_label] += 1

    return accuracy
    

def main():
    patterns, labels = load_iris(return_X_y=True)

    X_tr, X_te, y_tr, y_te = train_test_split(patterns, labels, test_size=0.3, stratify=labels)

    k = 7

    tr_acc = knn(k, X_tr, y_tr, X_tr, y_tr) # Train-set
    print(f'Class accuracy: {tr_acc} | Train-set accuracy: {np.round(np.sum(tr_acc) / len(X_tr), 2)}')

    te_acc = knn(k, X_tr, y_tr, X_te, y_te) # Test-set
    print(f'Class accuracy: {te_acc} | Test-set accuracy: {np.round(np.sum(te_acc) / len(X_te), 2)}')


if __name__ == '__main__':
    main()