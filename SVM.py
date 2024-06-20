__author__ = "Semin Kim"
__id__ = "2022312990"

# Do not import and other Python libraries
import numpy as np
import matplotlib.pyplot as plt


# Write your code following the instructions
class SVMClassifier:
    def __init__(self, n_iters=100, lr = 0.0001, random_seed=3, lambda_param=0.01):
        self.author = __author__
        self.id = __id__
        self.n_iters = n_iters # number of iterations
        self.lr = lr  # learning rate
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        np.random.seed(self.random_seed)


    def squared_hinge_loss(self, w,b,x,y):
        return max(0, 1 - y * (w @ x + b)) ** 2


    def fit(self, x, y):

        n_samples, n_features = x.shape

        # hint: in order to use y for SVM, change zeros to -1.
        y_ = y.copy()
        y_[np.where(y_ == 0)] = -1
        
        # hint: reset w, a numpy array with random values between 0 to 1, with the size of (n_features, ).
        init_w = np.random.uniform(0, 1, size=n_features)
        self.w = init_w
        self.b = 0 # reset b

        for _ in range(self.n_iters):
            for i in range(n_samples):
                x_i = x[i]
                y_i = y_[i]

                # hint: filter cases with y(i) * (w · x(i) + b) >= 1 using if statement.
                squared_hinge_loss = self.squared_hinge_loss(self.w, self.b, x_i, y_i)
                condition = squared_hinge_loss <= 0

                if condition:
                    # hint: update W using the Gradient Loss Function equation.
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # hint: update W using the Gradient Loss Function equation.
                    self.w -= self.lr * (2 * self.lambda_param * self.w - 
                                        2 * y_i * (1 - y_i * (self.w @ x_i + self.b)) * x_i)

                    self.b -= self.lr * (-2 * y_i * (1 - y_i * (self.w @ x_i + self.b)))

        return self


    def predict(self, x):

        predictions = x @ self.w + self.b

        return predictions


    def get_accuracy(self, y_true, y_pred):

        return np.mean(y_true == (y_pred >= 0))

