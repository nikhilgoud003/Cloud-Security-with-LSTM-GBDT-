

# Install required packages
!pip install xgboost
!pip install tensorflow
!pip install matplotlib seaborn scikit-learn
!pip install plotly
!pip install imbalanced-learn

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, precision_recall_curve,
                            average_precision_score, f1_score)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
import time

# Set styles for better visualizations
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Configure plots for better readability
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Starting enhanced threat detection system implementation...")

# Generate Synthetic Dataset with 45% threat ratio
def generate_data(samples=5000, threat_ratio=0.45):
    np.random.seed(42)
    timestamps = np.arange(samples)

    # Create base features
    features = np.random.randn(samples, 5)

    # Create more sophisticated threat signal
    periodic_signal = np.sin(timestamps / 50)  # Faster oscillation
    spike_signal = (np.random.rand(samples) > 0.95).astype(float) * 2  # Random spikes
    drift_signal = np.linspace(0, 1, samples)  # Slow drift

    # Combine signals to create threat pattern
    combined_signal = periodic_signal * 0.6 + spike_signal * 0.3 + drift_signal * 0.1
    noise = np.random.randn(samples) * 0.2

    # Threshold to get about 45% threats
    threshold = np.percentile(combined_signal + noise, 100 - (threat_ratio * 100))
    threat_signal = (combined_signal + noise) > threshold

    # Make threats more persistent (real threats often last multiple timesteps)
    for i in range(1, len(threat_signal)):
        if threat_signal[i-1] and np.random.rand() < 0.7:
            threat_signal[i] = True

    # Add threat signatures to features
    features[threat_signal, 0] += 1.5  # Strong signature in feature 0
    features[threat_signal, 1] -= 0.8  # Negative correlation in feature 1
    features[threat_signal, 2] += np.random.randn(threat_signal.sum()) * 0.5  # Noisy increase

    labels = threat_signal.astype(int)
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(5)])
    df['threat'] = labels

    # Add derived features
    df['feature_5'] = df['feature_0'] * df['feature_1']  # Interaction term
    df['feature_6'] = df['feature_2'].rolling(5).mean().fillna(0)  # Rolling average
    df['feature_7'] = df['feature_3'].diff().abs().fillna(0)  # Absolute difference

    return df

df = generate_data()
print("\nDataset preview:")
print(df.head())



plt.figure(figsize=(10, 6))

# Ensure 'threat' is integer type
df['threat'] = df['threat'].astype(int)

# Count values and compute percentages
class_counts = df['threat'].value_counts().sort_index()
total = len(df)
percentages = class_counts / total * 100

# Create the count plot with 'hue' set and legend disabled
ax = sns.countplot(x='threat', hue='threat', data=df, palette={0: '#3498db', 1: '#e74c3c'}, legend=False)

# Set title and labels
plt.title(f'Class Distribution (Threat: {percentages[1]:.1f}%)', fontsize=16, fontweight='bold')
plt.xlabel('Class (0: Safe, 1: Threat)', fontsize=13)
plt.ylabel('Count', fontsize=13)
plt.xticks([0, 1], ['0', '1'], fontsize=12)
plt.yticks(fontsize=12)

# Annotate bar values
for i, count in enumerate(class_counts):
    pct = percentages[i]
    ax.text(i, count + total * 0.02, f"{count} ({pct:.1f}%)",
            ha='center', fontsize=12, fontweight='bold', color='black')

plt.tight_layout()
plt.show()

# Add count labels on top of bars
for i, count in enumerate(class_counts):
    ax.text(i, count + 50, f"{count} ({count/len(df):.1%})",
            ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Feature engineering and scaling
scaler = StandardScaler()
features = [f'feature_{i}' for i in range(8)]
df[features] = scaler.fit_transform(df[features])

# Visualize feature distributions by class
plt.figure(figsize=(18, 12))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.kdeplot(df[df['threat'] == 0][feature], label='Safe', shade=True, color='#3498db')
    sns.kdeplot(df[df['threat'] == 1][feature], label='Threat', shade=True, color='#e74c3c')
    plt.title(f'{feature} Distribution by Class', fontsize=12)
    plt.xlabel(f'{feature} Value')
    plt.legend()
plt.tight_layout()
plt.show()

# Prepare Data for LSTM with improved windowing
def create_lstm_sequences(df, window=15):
    sequences = []
    targets = []
    for i in range(len(df) - window):
        seq_features = df.iloc[i:i+window][features].values
        # Add temporal differences as additional features
        diffs = np.diff(seq_features, axis=0)
        padded_diffs = np.vstack([np.zeros(seq_features.shape[1]), diffs])
        seq_features = np.hstack([seq_features, padded_diffs])

        sequences.append(seq_features)
        targets.append(df.iloc[i+window]['threat'])
    return np.array(sequences), np.array(targets)

print("\nPreparing sequence data for LSTM...")
start_time = time.time()
X_seq, y_seq = create_lstm_sequences(df)
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# Apply SMOTE only to training data to handle imbalance
smote = SMOTE(random_state=42)
X_train_seq_reshaped = X_train_seq.reshape(X_train_seq.shape[0], -1)
X_train_res, y_train_res = smote.fit_resample(X_train_seq_reshaped, y_train_seq)
X_train_seq = X_train_res.reshape(X_train_res.shape[0], X_train_seq.shape[1], X_train_seq.shape[2])

print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
print(f"\nLSTM data shape: {X_seq.shape}, {y_seq.shape}")
print(f"Training data: {X_train_seq.shape}, {y_train_res.shape}")
print(f"Testing data: {X_test_seq.shape}, {y_test_seq.shape}")

# Enhanced LSTM Model
print("\nBuilding enhanced LSTM model...")
lstm_model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

early_stopping = EarlyStopping(monitor='val_recall', patience=5, mode='max', restore_best_weights=True)

print("\nLSTM Model Summary:")
lstm_model.summary()

# Train the LSTM model
print("\nTraining LSTM model...")
start_time = time.time()
history = lstm_model.fit(
    X_train_seq, y_train_res,
    epochs=15,
    batch_size=64,
    validation_data=(X_test_seq, y_test_seq),
    callbacks=[early_stopping],
    verbose=1
)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Evaluate LSTM model
lstm_val_loss, lstm_val_acc, lstm_val_precision, lstm_val_recall = lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"\nLSTM Model - Validation Loss: {lstm_val_loss:.4f}, Accuracy: {lstm_val_acc:.4f}")
print(f"Precision: {lstm_val_precision:.4f}, Recall: {lstm_val_recall:.4f}")

# Generate LSTM predictions
lstm_probs = lstm_model.predict(X_test_seq).flatten()

# Find optimal threshold for threat detection
precision, recall, thresholds = precision_recall_curve(y_test_seq, lstm_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold for threat detection: {optimal_threshold:.4f}")
lstm_preds = (lstm_probs >= optimal_threshold).astype(int)

# Enhanced GBDT model with LSTM features
print("\nPreparing data for XGBoost model...")
X_gbdt = X_test_seq[:, -1, :8]  # Last timestep original features
lstm_features = X_test_seq.reshape(X_test_seq.shape[0], -1)  # All sequence features
X_gbdt = np.hstack([X_gbdt, lstm_features, lstm_probs.reshape(-1, 1)])

print(f"XGBoost input shape: {X_gbdt.shape}")

# Train XGBoost with class weighting
print("\nTraining XGBoost model...")
start_time = time.time()
gbdt_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train_res[y_train_res==0])/len(y_train_res[y_train_res==1]),
    eval_metric='aucpr',
    early_stopping_rounds=10
)

# Cross-validation
cv = StratifiedKFold(n_splits=3)
cv_scores = []
for train_idx, val_idx in cv.split(X_gbdt, y_test_seq):
    X_train, X_val = X_gbdt[train_idx], X_gbdt[val_idx]
    y_train, y_val = y_test_seq[train_idx], y_test_seq[val_idx]

    gbdt_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    y_pred = gbdt_model.predict(X_val)
    cv_scores.append(f1_score(y_val, y_pred))

print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1: {np.mean(cv_scores):.4f}")
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Final predictions
y_prob = gbdt_model.predict_proba(X_gbdt)[:, 1]
y_pred = (y_prob >= optimal_threshold).astype(int)

# Enhanced evaluation
print("\nEnhanced Classification Report:")
print(classification_report(y_test_seq, y_pred, target_names=['Safe', 'Threat']))

# Feature importance
feature_names = [f'feature_{i}' for i in range(8)] + \
               [f'seq_feat_{i}' for i in range(lstm_features.shape[1])] + \
               ['lstm_prob']
feature_importance = gbdt_model.feature_importances_

# Get top 20 features
top_features = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}) \
               .sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 20 Feature Importances', fontsize=16)
plt.tight_layout()
plt.show()

# Threat detection performance visualization
def plot_performance(y_true, y_pred, y_prob):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title('ROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    axes[2].plot(recall, precision, label=f'AP = {avg_precision:.2f}')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

plot_performance(y_test_seq, y_pred, y_prob)

# Threat timeline visualization
def plot_threat_timeline(y_true, y_pred, window=100):
    plt.figure(figsize=(15, 6))

    # Plot ground truth
    plt.plot(y_true[:window], 'o-', label='Actual', color='#2ecc71', alpha=0.7)

    # Plot predictions
    plt.plot(y_pred[:window], 'x-', label='Predicted', color='#e74c3c', alpha=0.7)

    # Highlight correct threat detections
    correct_threats = np.where((y_true[:window] == 1) & (y_pred[:window] == 1))[0]
    plt.scatter(correct_threats, y_true[correct_threats],
               color='#f1c40f', s=100, label='Correct Threat', zorder=3)

    plt.title('Threat Detection Timeline (First 100 Samples)')
    plt.xlabel('Time Step')
    plt.ylabel('Threat Status')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_threat_timeline(y_test_seq, y_pred)

from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Calculate metrics
f1 = f1_score(y_test_seq, y_pred)
precision = lstm_val_precision
recall = lstm_val_recall
accuracy = lstm_val_acc

# Create a 2x2 matrix of metrics
metrics_data = np.array([
    [accuracy, precision],
    [recall, f1]
])

# Create labels for the matrix
row_labels = ['Accuracy/Precision', 'Recall/F1']
col_labels = ['', '']  # Empty for cleaner look

# Create dataframe
metrics_matrix = pd.DataFrame(metrics_data,
                             index=row_labels,
                             columns=col_labels)

# Plot enhanced heatmap
plt.figure(figsize=(6, 4))
ax = sns.heatmap(metrics_matrix, annot=True, cmap='Blues', fmt='.3f',
                cbar=False, linewidths=2, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})

# Add metric names inside cells
ax.text(0.25, 0.25, 'Accuracy', ha='center', va='center', color='black', fontsize=10)
ax.text(1.25, 0.25, 'Precision', ha='center', va='center', color='black', fontsize=10)
ax.text(0.25, 1.25, 'Recall', ha='center', va='center', color='black', fontsize=10)
ax.text(1.25, 1.25, 'F1 Score', ha='center', va='center', color='black', fontsize=10)

# Customize appearance
plt.title('Model Evaluation Metrics Matrix',
          fontsize=16, fontweight='bold', pad=20)
plt.xticks([])
plt.yticks(rotation=0, fontsize=12)

# Add borders
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.show()
