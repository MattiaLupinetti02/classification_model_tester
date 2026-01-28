
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from classification_model_tester import ModelTester
from sklearn.datasets import load_iris

#load the dataset
data = load_iris()
X = data.data
y = data.target

data = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
data["target"] = y

# Define the target
target = "target"
# Define the models
models = {
    RandomForestClassifier(): {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    },
    SVC(): {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}

# Define Metrics
metrics = {
    'accuracy': accuracy_score,
    'f1': f1_score
}

# ModelTester object
tester = ModelTester(
    modelList=models,
    metrics=metrics,
    data=data,
    target=target,
    n_jobs=4
)

# Compute Best params 
tester.best_param_calculator(cv=5)

# Implement through the calculated hyperparameters
tester.implement_models(cv=5, metric='accuracy')

# plot learning curves and validation curves
tester.learning_curves()
tester.validation_curves()
