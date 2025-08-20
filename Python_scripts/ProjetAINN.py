import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset from CSV
data = pd.read_csv("data.csv")

# Split into features (X) and labels (y)
X = data[['T_Val', 'I_Val', 'U_Val']].values  # Replace with your actual column names
y = data[['T_Anomaly', 'I_Anomaly', 'U_Anomaly']].values  # Replace with your actual column names

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class AnomalyNN(nn.Module):
    def __init__(self):
        super(AnomalyNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer: 32 nodes
        self.fc2 = nn.Linear(64, 32)  # Hidden layer: 16 nodes
        self.fc3 = nn.Linear(32, 16)  # Input layer: 32 nodes
        self.fc4 = nn.Linear(16, 8)  # Hidden layer: 8 nodes
        self.fc5 = nn.Linear(8, 3)   # Output layer: 3 outputs

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # No activation here for BCEWithLogitsLoss
        return x

# Instantiate the model
model = AnomalyNN()
pos_weights = torch.tensor([1.0, 1.0, 1.0])  # Adjust weights based on class imbalance
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# Reduce learning rate for better convergence
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Import scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training loop with early stopping and learning rate scheduling
patience = 10  # Stop training if no improvement for 10 epochs
best_loss = float('inf')
patience_counter = 0

num_epochs = 1000
loss_values = []
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients
    logits = model(X_train)  # Forward pass
    loss = criterion(logits, y_train)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    
    # Step the scheduler with the current loss
    scheduler.step(loss.item())

    loss_values.append(loss.item())

    # Early stopping logic
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Test the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    logits = model(X_test)
    test_predictions = torch.round(torch.sigmoid(logits))  # Convert logits to binary predictions (0 or 1)
    print("Test Predictions:", test_predictions)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert predictions to binary numpy array
binary_predictions = test_predictions.numpy()

accuracy = accuracy_score(y_test.numpy(), binary_predictions)
precision = precision_score(y_test.numpy(), binary_predictions, average='macro', zero_division=0)
recall = recall_score(y_test.numpy(), binary_predictions, average='macro')
f1 = f1_score(y_test.numpy(), binary_predictions, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Plot the loss curve
import matplotlib.pyplot as plt

plt.plot(range(1, len(loss_values) + 1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()