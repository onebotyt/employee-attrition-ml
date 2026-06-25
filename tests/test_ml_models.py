import pytest
import numpy as np
import pandas as pd
from ml_models import (
    train_test_split, StandardScaler, LabelEncoder, SMOTE,
    LogisticRegression, RandomForestClassifier, KNeighborsClassifier, SVC,
    accuracy_score, confusion_matrix
)

# ── Utilities Tests ──

def test_label_encoder():
    le = LabelEncoder()
    data = np.array(["cat", "dog", "cat", "bird"])
    encoded = le.fit_transform(data)
    assert list(encoded) == [1, 2, 1, 0]
    assert list(le.classes_) == ["bird", "cat", "dog"]

def test_standard_scaler():
    sc = StandardScaler()
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaled = sc.fit_transform(data)
    # Mean should be 0
    assert np.allclose(scaled.mean(axis=0), [0.0, 0.0])
    # Std should be 1
    assert np.allclose(scaled.std(axis=0), [1.0, 1.0])

def test_train_test_split_stratified():
    X = np.arange(100).reshape(50, 2)
    y = np.array([0]*40 + [1]*10) # 80% 0s, 20% 1s
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    assert len(X_train) == 40
    assert len(X_test) == 10
    # Stratification check (should still be 80/20)
    assert np.mean(y_train == 1) == 0.20
    assert np.mean(y_test == 1) == 0.20

def test_smote_increases_minority():
    X = np.array([[1, 2], [1.5, 1.8], [2, 2], [8, 8], [8.5, 8.2]])
    y = np.array([0, 0, 0, 1, 1]) # Class 0 is majority (3), Class 1 is minority (2)
    
    sm = SMOTE(k_neighbors=1, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    assert len(y_res) == 6 # 3 of class 0, 3 of class 1
    assert np.sum(y_res == 0) == 3
    assert np.sum(y_res == 1) == 3

# ── Model Tests ──

def get_dummy_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # Simple linear decision boundary
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def test_logistic_regression_predict():
    X, y = get_dummy_data()
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
    assert accuracy_score(y, preds) > 0.8 # Should easily fit simple data

def test_random_forest_predict():
    X, y = get_dummy_data()
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

def test_knn_predict():
    X, y = get_dummy_data()
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

def test_svm_predict():
    X, y = get_dummy_data()
    model = SVC(kernel="linear", probability=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
    
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

# ── Metrics Tests ──

def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    # True negatives (0->0): 2
    # False positives (0->1): 1
    # False negatives (1->0): 0
    # True positives (1->1): 2
    assert cm[0, 0] == 2
    assert cm[0, 1] == 1
    assert cm[1, 0] == 0
    assert cm[1, 1] == 2
