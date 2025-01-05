# model_process.py

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    precision_score, roc_auc_score, accuracy_score
)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def gridSearch(model, param_grid, X_train, y_train, cv=2, scoring='accuracy', model_name='model_name'):
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid.fit(X_train, y_train)
    print("Best parameters for", model_name, ":", grid.best_params_)
    print("Best score for", model_name, ":", grid.best_score_)
    print("Best estimator for", model_name, ":", grid.best_estimator_)
    return grid

def randomForest(X_train, y_train,
                 n_estimators=100, max_depth=10, max_features='auto', 
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        max_features=max_features,
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        bootstrap=bootstrap
    )
    rf.fit(X_train, y_train)
    return rf

def xgboost_model(X_train, y_train,
                  n_estimators=100, max_depth=3, learning_rate=0.1,
                  objective='multi:softmax', random_state=42):
    xgb_clf = XGBClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate, 
        objective=objective, 
        random_state=random_state
    )
    xgb_clf.fit(X_train, y_train)
    return xgb_clf

class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=5, num_layers=1):
        """
        output_dim=5 for classes 0..4
        """
        super(PyTorchLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Final linear layer that outputs 5 logits
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) if batch_first=True
        batch_size = x.size(0)

        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_len, hidden_dim)
        
        # Take the output at the last time step
        out = out[:, -1, :]             # shape: (batch_size, hidden_dim)
        
        # Pass through final linear => produce logits for 5 classes
        logits = self.fc(out)           # shape: (batch_size, 5)
        return logits

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_pytorch_lstm(
    X_train, y_train,
    input_dim=581,
    hidden_dim=64,
    output_dim=5,
    num_layers=1,
    epochs=5,
    batch_size=16,
    lr=0.001,
    device='cpu',
    model_save_path='models/pytorch_lstm_model_classification.pth'
):
    """
    Train an LSTM for 5-class classification (labels 0..4).
    X_train: (num_samples, seq_len, input_dim)
    y_train: shape (num_samples,) with integer labels
    """
    # 1) Create Dataset & DataLoader
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)   # long for classification labels
    train_dataset = SequenceDataset(X_t, y_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2) Initialize model
    model = PyTorchLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()  # for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3) Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward
            logits = model(batch_X)  # shape: (batch_size, 5)
            loss = criterion(logits, batch_y)  # batch_y: (batch_size,)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    # 4) Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] LSTM classification model saved to {model_save_path}.")

    return model



def evaluate_model_classification(y_true, y_pred, average='macro'):
    """
    Compute classification metrics such as precision and (optionally) AUC.
    For multi-class, you can specify average='macro' or 'weighted' for precision, recall, etc.
    """
    prec = precision_score(y_true, y_pred, average=average)
    acc = accuracy_score(y_true, y_pred)
    return {
        "precision": prec,
        "accuracy": acc
    }

def evaluate_model_regression(y_true, y_pred):
    """
    Compute regression metrics such as MAE and RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}
