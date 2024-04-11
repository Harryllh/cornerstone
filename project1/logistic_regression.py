import numpy as np
import tqdm
import pdb

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda
        self.theta = None

    def fit(self, X, Y):
        """
            Train the logistic regression model using stochastic gradient descent.
        """
        self.theta = np.zeros((X.shape[1]))
        for epoch in range(self.num_epochs):
            indices = np.random.shuffle(np.arange(X.shape[0]))
            batch_ind = np.random.choice(indices, size=self.batch_size, replace=False)
            
            X_batch = X[batch_ind]
            Y_batch = Y[batch_ind]
            d_theta = self.gradient(X_batch, Y_batch)
            self.theta -= self.learning_rate * d_theta


    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """
        n = X.shape[0]
        y_pred = self.predict_proba(X)
        return (np.dot(X.T, (y_pred - Y)) + self.regularization_lambda * self.theta) / n


    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        prob = self.predict_proba(X)
        return 1 if prob > threshold else 0
    
    # def compute_loss(self, y_pred, y_true):
    #     m = y_true.shape[0]
    #     logprobs = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    #     return -np.sum(logprobs) / m

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))