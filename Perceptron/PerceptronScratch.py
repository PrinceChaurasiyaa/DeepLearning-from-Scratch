import numpy as np

def sigmoid(z):
    z = np.asarray(z)
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

def segmoid_derivative(z):
    s = sigmoid(z)
    return s * (1-s)

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class SinglePerceptron:
    def __init__ (self, n_features, learning_rate=0.01, epochs=100, random_state=42):
        self.lr = learning_rate
        self.epochs = epochs

        rng = np.random.default_rng(random_state)
        self.W = rng.normal(0, np.sqrt(1 / n_features), (n_features, 1))
        self.b = np.zeros((1, 1))

        self.history = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}

    def forward(self, X):
        self.z = X @ self.W + self.b
        self.a = sigmoid(self.z)
        return self.a

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)  # to make it column vector

        dz = self.a - y
        dW = (X.T @ dz) / m
        db = np.mean(dz)

        return dW, db

    def update(self, dW, db):
        self.W -= self.lr * dW
        self.b -= self.lr * db

    def predict_proba(self, X):
        return self.forward(X).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y.flatten())

    def fit(self, X_train, y_train, X_val, y_val, verbose=True):
        print("SINGLE PERCEPTRON (SCRATCH):")
        print("-"*69)
        print(f"  Features : {X_train.shape[1]}")
        print(f"  Train : {X_train.shape[0]} samples")
        print(f"  Val: {X_val.shape[0]} samples")
        print(f"  Learning Rate: {self.lr}  | Epochs : {self.epochs}")

        for epoch in range(1, self.epochs + 1):

            y_hat_train = self.forward(X_train)

            train_loss = binary_cross_entropy(y_train, y_hat_train)

            dW, db = self.backward(X_train, y_train)

            self.update(dW, db)

            y_hat_val = self.predict_proba(X_val)
            val_loss = binary_cross_entropy(y_val, y_hat_val)
            train_acc = self.accuracy(X_train, y_train)
            val_acc = self.accuracy(X_val, y_val)

            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(f"  Epoch {epoch:>4}/{self.epochs}"
                 f"  |  accuracy: {train_acc:.4f}"
                 f"  |  loss: {train_loss:.4f}"
                 f"  |  val_accuracy: {val_acc:.4f}"
                 f"  |  val_loss: {val_loss:.4f}")
            
        return self