import numpy as np
import matplotlib.pyplot as plt
from TfLinreg import TfLinreg

# We create test data set
X_train = np.arange(10).reshape(10, 1)
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

# We create and train the model
lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.01)
training_costs = lrmodel.train(X_train, y_train)

# We visualize the training costs after 10 epochs to check for convergence
plt.plot(range(1, len(training_costs) + 1), training_costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Training cost')
plt.show()

# We plot the linear regression fit on the training data
plt.scatter(X_train, y_train, marker='s', s=50, label='Training data')
plt.plot(range(X_train.shape[0]), lrmodel.predict(X_train), color='gray', marker='o', markersize=6, linewidth=3, label='Linreg model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()
