import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Multilayer_Perceptron import MyDataset, MLP
from DE import differential_evolution

def normalize(data):
    for x in data:
        x /= np.linalg.norm(x)

    return data


def compute_accuracy(model, dataloader):

    model = model.eval()

    correct = 0.0
    total_examples = 0

    for _, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            patterns = model(features)

        predictions = torch.argmax(patterns, dim=1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


def main():
    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Normalize data
    X = normalize(X)

    # Split the data into training and testing sets
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

    # Create validation test
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=1, stratify=y_tr)

    # Convert data to tensor datasets
    train_ds = MyDataset(X_tr, y_tr)
    val_ds = MyDataset(X_val, y_val)
    test_ds = MyDataset(X_te, y_te)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_ds, batch_size=32, shuffle=False,)

    # Training loop
    torch.manual_seed(1)

    # Model configuration
    num_features = X.shape[1]
    hidden_size = num_features * 2 - 1
    n_classes = len(np.unique(y))

    model = MLP(num_features, hidden_size, n_classes)

    # SIze of necesarry weights for the network
    n_weights = (num_features * hidden_size + hidden_size) + (hidden_size * n_classes + n_classes)
    
    num_epochs = 20

    torch.manual_seed(1)

    for epoch in range(num_epochs):

        model = model.train()
        for batch_idx, (patterns, labels) in enumerate(train_loader):

            outputs = model(patterns)

            model.zero_grad()

            # Obtain the weights by the metaheuristic
            weights = torch.tensor(differential_evolution(labels, outputs, n_weights, pop_size=len(labels)))

            # Ranges to separate the weights to fit the parameters
            size_w_layer1 = num_features * hidden_size
            size_bias_layer1 = hidden_size
            size_w_layer2 = hidden_size * n_classes

            # Create new paramaters using the values generated
            w_l1 = weights[:size_w_layer1]
            bias_l1 = weights[size_w_layer1:(size_w_layer1+size_bias_layer1)]
            w_l2 = weights[(size_w_layer1+size_bias_layer1):(size_w_layer1+size_bias_layer1+size_w_layer2)]
            bias_l2 = weights[(size_w_layer1+size_bias_layer1+size_w_layer2):]

            new_values = [w_l1, bias_l1, w_l2, bias_l2]

            # Assign new parameters
            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    if param.shape != new_values[i].shape:
                        param = torch.reshape(new_values[i], param.shape)
                    else:
                        param = new_values[i]

            ### LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}")
            
        print('-'*30)
                        

    # Evaluate results
    train_acc = compute_accuracy(model, train_loader)
    val_acc = compute_accuracy(model, val_loader)
    test_acc = compute_accuracy(model, test_loader)

    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")


if __name__ == '__main__':
    main()