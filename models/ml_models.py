# import standard libraries
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# import preprocessing file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing

# hyperparameters
seed = 1998
normalize = False
name_feature = False
downsampling_n_instances = 300
downsampling_n_instances_train = 400
downsampling_n_instances_test = 160

# load data
data_name = "HST"
data = DataPreprocessing(data_name, seed=seed)
train_datasets_HST, test_datasets_HST = data.load_data(
    downsampling_n_instances=downsampling_n_instances,
    normalize=normalize,
    name_feature=name_feature,
    convert_to_text=False,
    save_data=False,
)

data_name = "TEP"
data = DataPreprocessing(data_name, seed=seed)
train_datasets_TEP, test_datasets_TEP = data.load_data(
    downsampling_n_instances_train=downsampling_n_instances_train,
    downsampling_n_instances_test=downsampling_n_instances_test,
    extracted_label=[0, 1, 4, 5],
    normalize=normalize,
    name_feature=name_feature,
    convert_to_text=False,
    save_data=False,
)


# Extract data
X_train_TEP = train_datasets_TEP["X_train"]
y_train_TEP = train_datasets_TEP["y_train"]
X_test_TEP = test_datasets_TEP["X_test"]
y_test_TEP = test_datasets_TEP["y_test"]

# Optional: convert to numpy arrays
X_train_TEP = np.array(X_train_TEP)
y_train_TEP = np.array(y_train_TEP)
X_test_TEP = np.array(X_test_TEP)
y_test_TEP = np.array(y_test_TEP)

# Extract data
X_train_HST = train_datasets_HST["X_train"]
y_train_HST = train_datasets_HST["y_train"]
X_test_HST = test_datasets_HST["X_test"]
y_test_HST = test_datasets_HST["y_test"]

# Optional: convert to numpy arrays
X_train_HST = np.array(X_train_HST)
y_train_HST = np.array(y_train_HST)
X_test_HST = np.array(X_test_HST)
y_test_HST = np.array(y_test_HST)


# Models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


# Function to evaluate models
def evaluate_models(models, X_train, y_train, X_test, y_test, dataset_name):
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        results.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc,
            }
        )
    return results


# Run evaluation on both datasets
results_hst = evaluate_models(
    models, X_train_HST, y_train_HST, X_test_HST, y_test_HST, "HST"
)
results_tep = evaluate_models(
    models, X_train_TEP, y_train_TEP, X_test_TEP, y_test_TEP, "TEP"
)

# Combine and display results
comparison_df = pd.DataFrame(results_hst + results_tep)
print("comparison_df:", comparison_df)
