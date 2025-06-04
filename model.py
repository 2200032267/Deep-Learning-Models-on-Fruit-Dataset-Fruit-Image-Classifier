import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import io
import base64

# ✅ Load and Preprocess Data
def load_and_preprocess():
    file_path = "date_fruit_cleaned.csv"
    df = pd.read_csv(file_path)

    # ✅ Keep only numeric columns
    df = df.select_dtypes(include=['number'])

    # ✅ Remove missing values
    df.dropna(inplace=True)

    # ✅ Separate features (X) and target (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # ✅ Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Experiment 2: Data Loading and Preprocessing
def experiment_2_data_loading_and_preprocessing():
    file_path = "date_fruit_cleaned.csv"
    df = pd.read_csv(file_path)

    # Dataset shape and sample rows
    dataset_shape = df.shape
    sample_rows = df.head().to_dict()

    # Separate features (X) and target (y)
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Last column (Class)

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

    return {
        "dataset_shape": dataset_shape,
        "sample_rows": sample_rows,
        "normalized_values": X_normalized.tolist()[:5],  # First 5 normalized rows
        "encoded_labels": y_encoded.tolist()[:10],      # First 10 encoded labels
        "training_data_shape": X_train.shape,
        "testing_data_shape": X_test.shape,
    }

# ✅ Simple Neural Network (Multi-Class)
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)  # Output neurons = classes
        self.softmax = nn.Softmax(dim=1)  # Apply softmax for multiclass

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# ✅ Training Function (Multi-Class)
def train_simple_nn():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # ✅ Get number of classes dynamically
    num_classes = len(set(y_train))

    # ✅ Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # ✅ Define model, loss function, optimizer
    model = SimpleNN(input_size=X_train.shape[1], output_size=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Training Loop
    epochs = 10
    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    end_time = time.time()
    training_time = round(end_time - start_time, 2)

    # ✅ Evaluate Model
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predictions = torch.argmax(test_outputs, axis=1)

    # ✅ Performance Metrics
    accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)
    conf_matrix = confusion_matrix(y_test_tensor.numpy(), predictions.numpy()).tolist()
    class_report = classification_report(y_test_tensor.numpy(), predictions.numpy(), output_dict=True)

    return {
        "accuracy": round(accuracy * 100, 2),
        "loss_over_epochs": losses,
        "sample_predictions": predictions.numpy()[:10].tolist(),
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "training_time": training_time
    }

# ✅ Sequential Model for Multiclass Classification (Experiment 3)
def train_sequential_nn():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    num_classes = len(set(y_train))

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
        nn.Softmax(dim=1)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # ✅ Evaluate Model
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predictions = torch.argmax(test_outputs, axis=1)

    accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
    conf_matrix = confusion_matrix(y_test_tensor.numpy(), predictions.numpy())
    class_report = classification_report(y_test_tensor.numpy(), predictions.numpy(), output_dict=True)

    # Generate plots
    loss_plot_url = generate_plot(train_losses, "Loss Over Epochs", "Epochs", "Loss")
    cm_plot_url = generate_confusion_matrix_plot(conf_matrix)

    return {
        "accuracy": round(accuracy * 100, 2),
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
        "train_losses": train_losses,
        "loss_plot_url": loss_plot_url,
        "cm_plot_url": cm_plot_url,
    }

# ✅ Experiment 4: Sequential Model with Multiple Optimizers
def train_sequential_nn_with_optimizers():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    num_classes = len(set(y_train))

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    optimizers = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop
    }

    results = {}

    for opt_name, opt_class in optimizers.items():
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

        optimizer = opt_class(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        train_losses = []

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ✅ Evaluate Model
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predictions = torch.argmax(test_outputs, axis=1)

        accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
        conf_matrix = confusion_matrix(y_test_tensor.numpy(), predictions.numpy())
        class_report = classification_report(y_test_tensor.numpy(), predictions.numpy(), output_dict=True)

        # Generate plots
        loss_plot_url = generate_plot(train_losses, f"Loss Over Epochs ({opt_name})", "Epochs", "Loss")
        cm_plot_url = generate_confusion_matrix_plot(conf_matrix)

        results[opt_name] = {
            "accuracy": round(accuracy * 100, 2),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "loss_plot_url": loss_plot_url,
            "cm_plot_url": cm_plot_url,
        }

    return results

# ✅ Experiment 5: Random Mini-Batch Evaluations
def experiment_5_random_mini_batch():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, len(set(y_train))),
        nn.Softmax(dim=1)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epoch_losses = []
    epoch_accuracies = []
    mini_batch_accuracies = []

    for epoch in range(10):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate mini-batch accuracy
            predictions = torch.argmax(outputs, axis=1)
            accuracy = accuracy_score(batch_y.numpy(), predictions.numpy())
            mini_batch_accuracies.append(accuracy)

        epoch_losses.append(total_loss / len(dataloader))

        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1)
            accuracy = accuracy_score(y_test, y_pred)
            epoch_accuracies.append(accuracy * 100)

    # Generate plots
    loss_plot_url = generate_plot(epoch_losses, "Loss Over Epochs", "Epochs", "Loss")
    accuracy_plot_url = generate_plot(epoch_accuracies, "Accuracy Over Epochs", "Epochs", "Accuracy")
    mini_batch_accuracy_plot_url = generate_plot(mini_batch_accuracies, "Mini-Batch Accuracy Over Epochs", "Mini-Batch", "Accuracy")

    # Decision Boundary Plot (for 2D data)
    if X_train.shape[1] == 2:
        decision_boundary_plot_url = generate_decision_boundary_plot(model, X_train, y_train)
    else:
        decision_boundary_plot_url = None

    return {
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "mini_batch_accuracies": mini_batch_accuracies,
        "loss_plot_url": loss_plot_url,
        "accuracy_plot_url": accuracy_plot_url,
        "mini_batch_accuracy_plot_url": mini_batch_accuracy_plot_url,
        "decision_boundary_plot_url": decision_boundary_plot_url,
    }

# ✅ Helper Function to Generate Plots
def generate_plot(data, title, xlabel, ylabel):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(data) + 1), data, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ✅ Helper Function to Generate Confusion Matrix Plot
def generate_confusion_matrix_plot(cm):
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_decision_boundary_plot(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
        Z = torch.argmax(Z, axis=1).numpy()
        Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')