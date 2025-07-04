import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import random

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

# 1. Load data
df = pd.read_csv("cnn_ready_with_pheno.csv", index_col=0)
X = df.drop(columns=["phenotype"]).values.astype(np.float32)
y = df["phenotype"].values.astype(np.float32)

# 2. Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Train/test split (80/20) - matching CNN approach
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, shuffle=True
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}")

# 4. Initialize XGBoost model
# Using parameters optimized for genomic data
model = xgb.XGBRegressor(
    n_estimators=200,  # Number of boosting rounds
    max_depth=6,  # Control overfitting
    learning_rate=0.1,  # Step size shrinkage
    subsample=0.8,  # Row subsampling
    colsample_bytree=0.8,  # Column subsampling
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=seed,
    n_jobs=-1,  # Use all available cores
    verbosity=1,  # Show progress
)

# 5. Training
print("Starting training...")

# Train the model (simplified without early stopping for compatibility)
model.fit(X_train, y_train)

print("Training completed!")

# 6. Make predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# 7. Calculate metrics (matching CNN evaluation)
train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Convert to classification metrics (matching CNN approach)
# Using same threshold as CNN (0.5)
train_pred_labels = (train_preds > 0.5).astype(int)
train_true_labels = (y_train > 0.5).astype(int)
test_pred_labels = (test_preds > 0.5).astype(int)
test_true_labels = (y_test > 0.5).astype(int)

train_acc = accuracy_score(train_true_labels, train_pred_labels)
test_acc = accuracy_score(test_true_labels, test_pred_labels)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 8. Feature importance analysis (XGBoost provides multiple importance types)
feature_names = df.drop(columns=["phenotype"]).columns

# Get feature importance (gain-based)
feature_importance = pd.DataFrame(
    {"feature": feature_names, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

print("\nTop 10 most important features (by gain):")
print(feature_importance.head(10))

# 9. Additional XGBoost-specific metrics
print(f"\nTotal trees used: {model.n_estimators}")

# 10. Model performance summary
print("\n=== Model Performance Summary ===")
print("Model: XGBoost Regressor")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}")
print(f"Total estimators: {model.n_estimators}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 11. Optional: Plot feature importance (top 20)
try:
    import matplotlib.pyplot as plt

    # Plot top 20 features
    top_features = feature_importance.head(20)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features["importance"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Feature Importances (XGBoost)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("xgboost_feature_importance.png", dpi=300, bbox_inches="tight")
    print("Feature importance plot saved as 'xgboost_feature_importance.png'")
    plt.close()

except ImportError:
    print("Matplotlib not available, skipping feature importance plot")

# Optional: Save predictions for further analysis
predictions_df = pd.DataFrame(
    {
        "true_values": y_test,
        "predictions": test_preds,
        "predicted_labels": test_pred_labels,
        "true_labels": test_true_labels,
    }
)

# Uncomment to save predictions
# predictions_df.to_csv("xgboost_predictions.csv", index=False)
# print("Predictions saved to xgboost_predictions.csv")

# Optional: Save the trained model
# model.save_model("xgboost_model.json")
# print("Model saved as xgboost_model.json")
