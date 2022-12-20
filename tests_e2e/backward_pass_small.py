import numpy as np
from activations.relu import ReLU
from layers.linear import LinearLayer
from layers.sequential import Sequential
from loss.cross_entropy import SMCrossEntropyLoss
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(0)

    # Create a Sequential model
    model = Sequential(LinearLayer(4, 6), ReLU(), LinearLayer(6, 3))

    # Create a CrossEntropyLoss function
    loss = SMCrossEntropyLoss()

    # Create a dataset
    Data, labels = load_iris(return_X_y=True)
    # Normalize data
    # one hot encode the labels
    labels = np.eye(3)[labels]

    # Train the model
    loss_history = []
    for epoch in range(100):
        ls = 0
        for i, (x, y) in enumerate(zip(Data, labels)):
            x_out = model.forward(x)
            ls += loss(x_out, y)
            model.backward(loss.backward(x_out, y))
            model.update_params(0.01)
        ls /= len(Data)
        print("Epoch: {}, Loss: {}".format(epoch, ls))
        loss_history.append(ls)

    # Plot the loss on the y-axis and the epoch on the x-axis
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.show()
