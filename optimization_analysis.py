import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. NEURAL NETWORK CLASS
# ==========================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # He Initialization - good for tanh/sigmoid activations
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)  # Hidden layer activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # Output (probability)
        return self.a2

    def train(self, X, y, epochs, batch_size, lr):
        m = X.shape[0]
        loss_history = []

        for epoch in range(epochs):
            # SHUFFLING - Mix data before each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            batch_count = 0

            # MINI-BATCH LOOP
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                curr_batch_size = X_batch.shape[0]

                # Forward Propagation
                output = self.forward(X_batch)

                # Loss (Binary Cross-Entropy)
                loss = -np.mean(y_batch * np.log(output + 1e-8) + (1 - y_batch) * np.log(1 - output + 1e-8))
                epoch_loss += loss
                batch_count += 1

                # Backward Propagation
                dz2 = output - y_batch
                dW2 = np.dot(self.a1.T, dz2) / curr_batch_size
                db2 = np.sum(dz2, axis=0, keepdims=True) / curr_batch_size

                da1 = np.dot(dz2, self.W2.T) * (1 - np.power(self.a1, 2)) # tanh derivative
                dW1 = np.dot(X_batch.T, da1) / curr_batch_size
                db1 = np.sum(da1, axis=0, keepdims=True) / curr_batch_size

                # Weight Update
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            loss_history.append(epoch_loss / batch_count)
        return loss_history

# ==========================================
# 2. DATA PREPARATION (CHURN SIMULATION)
# ==========================================
X_raw, y_raw = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
y_raw = y_raw.reshape(-1, 1)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ==========================================
# 3. RUNNING THE EXPERIMENT
# ==========================================
epochs = 100
lr = 0.05
input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = 1

print("Starting optimization race...")

# Model 1: BATCH
nn_batch = NeuralNetwork(input_dim, hidden_dim, output_dim)
loss_batch = nn_batch.train(X_train, y_train, epochs, batch_size=len(X_train), lr=lr)
print("✔ Batch GD finished")

# Model 2: SGD
nn_sgd = NeuralNetwork(input_dim, hidden_dim, output_dim)
loss_sgd = nn_sgd.train(X_train, y_train, epochs, batch_size=1, lr=lr)
print("✔ Stochastic GD finished")

# Model 3: MINI-BATCH
nn_minibatch = NeuralNetwork(input_dim, hidden_dim, output_dim)
loss_minibatch = nn_minibatch.train(X_train, y_train, epochs, batch_size=32, lr=lr)
print("✔ Mini-batch GD finished")

# ==========================================
# 4. PLOTTING RESULTS (ENGLISH LABELS)
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(loss_batch, label='Batch (Stable/Slow)', color='red', linewidth=2)
plt.plot(loss_sgd, label='SGD (Noisy/Fast)', color='blue', alpha=0.3)
plt.plot(loss_minibatch, label='Mini-batch (Optimal)', color='green', linewidth=2)

plt.title('Gradient Descent Variants Comparison (Loss Convergence)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# TO JEST KLUCZOWE: to wymusza otwarcie okna z wykresem
plt.show()

# Final Accuracy check
def calculate_accuracy(model, X, y):
    predictions = (model.forward(X) > 0.5).astype(float)
    return np.mean(predictions == y) * 100

print(f"\nFinal Test Accuracy:")
print(f"Batch: {calculate_accuracy(nn_batch, X_test, y_test):.2f}%")
print(f"SGD: {calculate_accuracy(nn_sgd, X_test, y_test):.2f}%")
print(f"Mini-batch: {calculate_accuracy(nn_minibatch, X_test, y_test):.2f}%")
