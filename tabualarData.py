
#%%
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load the dataset

PROJECT_ROOT = Path(__file__).resolve().parent

data = pd.read_csv(str(PROJECT_ROOT) + '/Multimodal-Quiz/clinical_df.csv')

#%%
from my_utils.BF_utils import *

data = data.drop(["tertiary_gleason","BCR_PSA"], axis=1)
continuous_features, discrete_features, id_or_date = divide_features(data)

category_mappings3 = {}
for col in discrete_features:
    cat_series = data[col].astype('category')
    category_mappings3[col] = dict(enumerate(cat_series.cat.categories))
    data[col] = cat_series.cat.codes.replace(-1, np.nan)


# Splitting the dataset into features and target variable, adjust labels to be zero-indexed
X = data.drop(['event','duration'], axis=1)
missing_summary = X.isna().sum().sort_values(ascending=False)
print(missing_summary)
y = data['event']   # Adjusting labels to be zero-indexed
# %%
# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%
# Define the TabTransformer model
class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes, dim_embedding=64, num_heads=4, num_layers=4):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_embedding, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Adding a sequence length dimension
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Pooling
        x = self.classifier(x)
        return x
# %%

# Initialize the model, loss, and optimizer
# Initialize the model, loss, and optimizer
device = torch.device('cpu')
model = TabTransformer(X.shape[1], 2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to tensors AND move them to the device
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
#%%
# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    # Forward pass
    output = model(X_train_tensor)

    # Loss
    loss = criterion(output, y_train_tensor)

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# %%
# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model.eval()

# Move to device
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

with torch.no_grad():
    predictions = model(X_test_tensor)
    _, predicted_classes = torch.max(predictions, 1)

# Move back to CPU for sklearn metrics
y_true = y_test_tensor.cpu().numpy()
y_pred = predicted_classes.cpu().numpy()

# Metrics
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')   # or 'macro'
recall    = recall_score(y_true, y_pred, average='binary')
f1        = f1_score(y_true, y_pred, average='binary')

print("===== TEST METRICS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# %%
