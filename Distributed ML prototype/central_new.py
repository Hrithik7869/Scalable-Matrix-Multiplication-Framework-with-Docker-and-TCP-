import socket
import struct
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Config
WORKER_NODES = [("worker_1", 5000), ("worker_2", 5001)]

NUM_WEIGHTS = 10
EPOCHS = 10

# Define batch size
BATCH_SIZE = 1000

# Load dataset
df = pd.read_csv("housing.csv")
print("Columns in the dataset:", df.columns)

# Ensure required columns exist
assert "ocean_proximity" in df.columns, "'ocean_proximity' column is missing!"
assert "median_house_value" in df.columns, "'median_house_value' column is missing!"

# Preprocess dataset
X = df.drop(columns=["ocean_proximity", "median_house_value"]).astype(np.float32)
y = df["median_house_value"].astype(np.float32)

# Handle missing values in features
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Ensure no missing values remain
assert not X.isnull().values.any(), "Features contain NaN values!"
assert not y.isnull().values.any(), "Target contains NaN values!"

# Pad features if necessary
if X.shape[1] < NUM_WEIGHTS:
    for i in range(NUM_WEIGHTS - X.shape[1]):
        X[f"pad_{i}"] = 0.1 * i
X = X.iloc[:, :NUM_WEIGHTS]  # Ensure exactly NUM_WEIGHTS

# Normalize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
x_train_1, x_train_2 = np.array_split(x_train, 2)
y_train_1, y_train_2 = np.array_split(y_train, 2)

print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
print(f"x_train_1 shape: {x_train_1.shape}, x_train_2 shape: {x_train_2.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
print(f"y_train_1 shape: {y_train_1.shape}, y_train_2 shape: {y_train_2.shape}")

# Initialize weights
weights = np.random.uniform(-0.01, 0.01, NUM_WEIGHTS).astype(np.float32)

def serialize_dataset(x, y):
    data = struct.pack('I', len(x))  # Send number of samples first
    for i in range(len(x)): 
        row = x.iloc[i].values
        label = y.iloc[i]
        for val in row:
            data += struct.pack('f', val)
        data += struct.pack('f', label)
    return data

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Worker disconnected")
        data += packet
    return data

def communicate_with_worker(ip, port, weights, x_chunk, y_chunk):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(struct.pack('I', len(x_chunk)))  # ✅ Send number of samples
        sock.sendall(weights.tobytes())
        sock.sendall(serialize_dataset(x_chunk, y_chunk))
        updated = recvall(sock, weights.nbytes)
        return np.frombuffer(updated, dtype=np.float32)

# Helper function to create batches
def create_batches(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x.iloc[i:i + batch_size], y.iloc[i:i + batch_size]

# Training loop with batch processing
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch:3d} - Starting Batch Training")
    
    # Create batches for both worker nodes
    batches_1 = list(create_batches(x_train_1, y_train_1, BATCH_SIZE))
    batches_2 = list(create_batches(x_train_2, y_train_2, BATCH_SIZE))
    
    # Ensure both workers have the same number of batches
    num_batches = min(len(batches_1), len(batches_2))
    
    for batch_idx in range(num_batches):
        print(f"Processing Batch {batch_idx + 1}/{num_batches}")
        
        # Get the current batch for each worker
        x_batch_1, y_batch_1 = batches_1[batch_idx]
        x_batch_2, y_batch_2 = batches_2[batch_idx]
        
        # Send batch to worker nodes and receive updated weights
        updated_weights_1 = communicate_with_worker(
            WORKER_NODES[0][0], WORKER_NODES[0][1], weights, x_batch_1, y_batch_1
        )
        updated_weights_2 = communicate_with_worker(
            WORKER_NODES[1][0], WORKER_NODES[1][1], weights, x_batch_2, y_batch_2
        )
        
        # Aggregate weights
        n1, n2 = len(x_batch_1), len(x_batch_2)
        weights = (updated_weights_1 * n1 + updated_weights_2 * n2) / (n1 + n2)
        

        # Send aggregated weights back to workers
        for worker in WORKER_NODES:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(worker)
                sock.sendall(weights.tobytes())

        
        print(f"Batch {batch_idx + 1}/{num_batches} - Aggregated Weights:")
        print(" ".join(f"{w:7.4f}" for w in weights))
    
    print(f"Epoch {epoch:3d} - Completed")

# Evaluate the model
model = LinearRegression()

if len(weights) == x_test.shape[1] + 1:
    model.coef_ = weights[:-1]
    model.intercept_ = weights[-1]
else:
    model.coef_ = weights
    model.intercept_ = 0

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error on Test Data: {mse:.4f}")
