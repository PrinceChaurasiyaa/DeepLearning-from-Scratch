from LSTMScratch import LSTM
import numpy as np


print("=" * 50)
print("LSTM from Scratch - Example Usage")
print("=" * 50)

# Example 1: Simple sequence prediction
print("\n1. Training on sine wave prediction:")

# Generate sine wave data
seq_len = 20
n_samples = 100

X_train = []
y_train = []

for _ in range(n_samples):
    t = np.linspace(0, 2*np.pi, seq_len + 1)
    sine = np.sin(t)
    X_train.append(sine[:-1].reshape(-1, 1))
    y_train.append(sine[1:].reshape(-1, 1))

# Initialize LSTM
lstm = LSTM(input_size=1, hidden_size=32, output_size=1, learning_rate=0.01)

# Training
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for i in range(n_samples):
        X = np.array(X_train[i]).reshape(seq_len, 1, 1)
        y = np.array(y_train[i]).reshape(seq_len, 1, 1)
        loss = lstm.train_step(X, y)
        total_loss += loss

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_samples:.6f}")

# Test prediction
print("\n2. Testing prediction:")
test_input = X_train[0].reshape(seq_len, 1, 1)
prediction = lstm.predict(test_input, return_sequences=True)

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {prediction.shape}")
print(f"Sample predictions: {prediction[:5].flatten()}")
print(f"Sample targets: {y_train[0][:5].flatten()}")
