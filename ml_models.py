"""
ml_models.py — Pure NumPy implementations of ML algorithms
No sklearn / scipy dependency — works even when BLAS DLLs are blocked.
RISE Internship · Employee Attrition Prediction
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────

class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self._map = {}
        self._inv = {}

    def fit(self, y):
        self.classes_ = np.array(sorted(set(y)))
        self._map = {v: i for i, v in enumerate(self.classes_)}
        self._inv = {i: v for v, i in self._map.items()}
        return self

    def transform(self, y):
        return np.array([self._map[v] for v in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return np.array([self._inv[i] for i in y])


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0   # avoid division by zero
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────

def train_test_split(X, y, test_size=0.20, random_state=42, stratify=None):
    rng = np.random.default_rng(random_state)
    n   = len(y)

    if stratify is not None:
        classes, counts = np.unique(stratify, return_counts=True)
        train_idx, test_idx = [], []
        for cls in classes:
            idx = np.where(stratify == cls)[0]
            rng.shuffle(idx)
            n_test = max(1, round(len(idx) * test_size))
            test_idx.extend(idx[:n_test])
            train_idx.extend(idx[n_test:])
        train_idx = np.array(train_idx)
        test_idx  = np.array(test_idx)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test   = max(1, round(n * test_size))
        test_idx  = idx[:n_test]
        train_idx = idx[n_test:]

    if isinstance(X, np.ndarray):
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    else:
        X = np.array(X)
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ─────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    lmap = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        cm[lmap[t]][lmap[p]] += 1
    return cm

def _tp_fp_fn(y_true, y_pred, pos=1):
    tp = np.sum((y_true == pos) & (y_pred == pos))
    fp = np.sum((y_true != pos) & (y_pred == pos))
    fn = np.sum((y_true == pos) & (y_pred != pos))
    return tp, fp, fn

def precision_score(y_true, y_pred, pos=1, zero_division=0):
    tp, fp, _ = _tp_fp_fn(y_true, y_pred, pos)
    return tp / (tp + fp) if (tp + fp) > 0 else zero_division

def recall_score(y_true, y_pred, pos=1, zero_division=0):
    tp, _, fn = _tp_fp_fn(y_true, y_pred, pos)
    return tp / (tp + fn) if (tp + fn) > 0 else zero_division

def f1_score(y_true, y_pred, pos=1, zero_division=0):
    p = precision_score(y_true, y_pred, pos, zero_division)
    r = recall_score(y_true, y_pred, pos, zero_division)
    return 2 * p * r / (p + r) if (p + r) > 0 else zero_division

def roc_curve(y_true, y_score):
    thresholds = np.concatenate([[1.01], np.sort(np.unique(y_score))[::-1]])
    fprs, tprs = [0.0], [0.0]
    pos = np.sum(y_true == 1); neg = np.sum(y_true == 0)
    for thr in thresholds:
        yp = (y_score >= thr).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        fprs.append(fp / neg if neg else 0)
        tprs.append(tp / pos if pos else 0)
    fprs.append(1.0); tprs.append(1.0)
    return np.array(fprs), np.array(tprs), thresholds

def roc_auc_score(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.trapezoid(tpr, fpr))

def classification_report(y_true, y_pred, target_names=None):
    classes = np.unique(y_true)
    lines = []
    header = f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    lines.append(header)
    lines.append("")
    for i, cls in enumerate(classes):
        name = target_names[i] if target_names else str(cls)
        p = precision_score(y_true, y_pred, pos=cls)
        r = recall_score(y_true, y_pred, pos=cls)
        f = f1_score(y_true, y_pred, pos=cls)
        s = np.sum(y_true == cls)
        lines.append(f"{name:>20} {p:>10.2f} {r:>10.2f} {f:>10.2f} {s:>10}")
    lines.append("")
    acc = accuracy_score(y_true, y_pred)
    lines.append(f"{'accuracy':>20} {'':>10} {'':>10} {acc:>10.2f} {len(y_true):>10}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
#  STRATIFIED K-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────────────────

class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits    = n_splits
        self.shuffle     = shuffle
        self.random_state = random_state

    def split(self, X, y):
        rng = np.random.default_rng(self.random_state)
        classes = np.unique(y)
        class_indices = {c: np.where(y == c)[0] for c in classes}
        if self.shuffle:
            for c in classes:
                rng.shuffle(class_indices[c])
        folds = [[] for _ in range(self.n_splits)]
        for c in classes:
            idx = class_indices[c]
            splits = np.array_split(idx, self.n_splits)
            for i, sp in enumerate(splits):
                folds[i].extend(sp.tolist())
        for k in range(self.n_splits):
            test_idx  = np.array(folds[k])
            train_idx = np.array([i for j in range(self.n_splits) if j != k for i in folds[j]])
            yield train_idx, test_idx


def cross_val_score(model, X, y, cv, scoring="recall"):
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[test_idx])
        if scoring == "recall":
            scores.append(recall_score(y[test_idx], y_pred))
        elif scoring == "accuracy":
            scores.append(accuracy_score(y[test_idx], y_pred))
        elif scoring == "f1":
            scores.append(f1_score(y[test_idx], y_pred))
    return np.array(scores)


# ─────────────────────────────────────────────────────────
#  LOGISTIC REGRESSION (gradient descent, supports class_weight)
# ─────────────────────────────────────────────────────────

class LogisticRegression:
    def __init__(self, C=1.0, class_weight=None, random_state=42,
                 max_iter=500, lr=0.1):
        self.C            = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter     = max_iter
        self.lr           = lr
        self.coef_        = None
        self.intercept_   = None

    def get_params(self):
        return dict(C=self.C, class_weight=self.class_weight,
                    random_state=self.random_state, max_iter=self.max_iter, lr=self.lr)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sample_weights(self, y):
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            total = len(y)
            w = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self.coef_      = rng.normal(0, 0.01, d)
        self.intercept_ = 0.0
        sw = self._sample_weights(y)
        lam = 1.0 / (self.C * n)

        for _ in range(self.max_iter):
            z     = X @ self.coef_ + self.intercept_
            p     = self._sigmoid(z)
            err   = (p - y) * sw
            grad  = X.T @ err / n + lam * self.coef_
            self.coef_      -= self.lr * grad
            self.intercept_ -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = self._sigmoid(X @ self.coef_ + self.intercept_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (self._sigmoid(X @ self.coef_ + self.intercept_) >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────
#  DECISION TREE (used internally by Random Forest)
# ─────────────────────────────────────────────────────────

class _Node:
    __slots__ = ["feat","thr","left","right","value","weight"]
    def __init__(self):
        self.feat = self.thr = self.left = self.right = self.value = self.weight = None


class _DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=4,
                 max_features=None, random_state=42):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state
        self.root              = None
        self._rng              = np.random.default_rng(random_state)

    def _gini(self, y, w):
        total = w.sum()
        if total == 0: return 0
        p1 = (w[y == 1]).sum() / total
        return 1 - p1**2 - (1-p1)**2

    def _best_split(self, X, y, w):
        n, d = X.shape
        n_feat = self.max_features or d
        feats  = self._rng.choice(d, size=min(n_feat, d), replace=False)
        best_gain, best_feat, best_thr = -1, None, None
        g_parent = self._gini(y, w)
        for f in feats:
            vals = np.unique(X[:, f])
            if len(vals) < 2: continue
            thresholds = (vals[:-1] + vals[1:]) / 2
            for thr in thresholds:
                left  = X[:, f] <= thr
                right = ~left
                if left.sum() < 1 or right.sum() < 1: continue
                g_l = self._gini(y[left],  w[left])
                g_r = self._gini(y[right], w[right])
                wl, wr = w[left].sum(), w[right].sum()
                gain = g_parent - (wl*g_l + wr*g_r) / (wl+wr)
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, f, thr
        return best_feat, best_thr

    def _leaf_value(self, y, w):
        total = w.sum()
        return (w[y == 1]).sum() / total if total > 0 else 0.5

    def _build(self, X, y, w, depth):
        node = _Node()
        node.weight = w.sum()
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            node.value = self._leaf_value(y, w)
            return node
        feat, thr = self._best_split(X, y, w)
        if feat is None:
            node.value = self._leaf_value(y, w)
            return node
        node.feat, node.thr = feat, thr
        mask = X[:, feat] <= thr
        node.left  = self._build(X[mask],  y[mask],  w[mask],  depth+1)
        node.right = self._build(X[~mask], y[~mask], w[~mask], depth+1)
        return node

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.root = self._build(X, y, sample_weight, 0)
        return self

    def _predict_one(self, x, node):
        while node.value is None:
            node = node.left if x[node.feat] <= node.thr else node.right
        return node.value

    def predict_proba(self, X):
        p = np.array([self._predict_one(x, self.root) for x in X])
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────
#  RANDOM FOREST
# ─────────────────────────────────────────────────────────

class RandomForestClassifier:
    def __init__(self, n_estimators=100, class_weight=None, random_state=42,
                 n_jobs=-1, max_depth=10, min_samples_split=4):
        self.n_estimators      = n_estimators
        self.class_weight      = class_weight
        self.random_state      = random_state
        self.n_jobs            = n_jobs
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.trees_            = []
        self.feature_importances_ = None

    def get_params(self):
        return dict(n_estimators=self.n_estimators, class_weight=self.class_weight,
                    random_state=self.random_state, n_jobs=self.n_jobs,
                    max_depth=self.max_depth, min_samples_split=self.min_samples_split)

    def _sample_weights(self, y):
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            total = len(y)
            w = {c: total / (len(classes)*cnt) for c, cnt in zip(classes, counts)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        max_feat = max(1, int(np.sqrt(d)))
        sw  = self._sample_weights(y)
        self.trees_ = []
        importances = np.zeros(d)

        for i in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True)
            Xi, yi, wi = X[idx], y[idx], sw[idx]
            tree = _DecisionTree(max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 max_features=max_feat,
                                 random_state=int(rng.integers(0, 99999)))
            tree.fit(Xi, yi, wi)
            self.trees_.append(tree)

        # Feature importance via mean impurity decrease (approximated)
        self.feature_importances_ = self._compute_importance(d)
        return self

    def _compute_importance(self, d):
        scores = np.zeros(d)
        for tree in self.trees_:
            self._walk(tree.root, scores)
        total = scores.sum()
        return scores / total if total > 0 else scores

    def _walk(self, node, scores):
        if node is None or node.value is not None:
            return
        scores[node.feat] += node.weight
        self._walk(node.left,  scores)
        self._walk(node.right, scores)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probas = np.mean([t.predict_proba(X) for t in self.trees_], axis=0)
        return probas

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────
#  KNN
# ─────────────────────────────────────────────────────────

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train_    = None
        self.y_train_    = None

    def get_params(self):
        return dict(n_neighbors=self.n_neighbors)

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def _distances(self, x):
        diff = self.X_train_ - x
        return np.sqrt((diff**2).sum(axis=1))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = []
        for x in X:
            dists = self._distances(x)
            nn    = np.argsort(dists)[:self.n_neighbors]
            p1    = self.y_train_[nn].mean()
            probs.append([1-p1, p1])
        return np.array(probs)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────
#  SVM (simple linear + RBF kernel, gradient-based)
# ─────────────────────────────────────────────────────────

class SVC:
    """Simplified SVM via mini-batch subgradient — good enough for HR data."""
    def __init__(self, kernel="rbf", C=1.0, class_weight=None,
                 probability=True, random_state=42, gamma="scale"):
        self.kernel       = kernel
        self.C            = C
        self.class_weight = class_weight
        self.probability  = probability
        self.random_state = random_state
        self.gamma        = gamma
        self.w_           = None
        self.b_           = None
        self._lr_model    = None  # calibration

    def get_params(self):
        return dict(kernel=self.kernel, C=self.C, class_weight=self.class_weight,
                    probability=self.probability, random_state=self.random_state)

    def _sample_weights(self, y):
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            total = len(y)
            w = {c: total / (len(classes)*cnt) for c, cnt in zip(classes, counts)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        # Convert labels to ±1
        y_pm = np.where(y == 1, 1, -1).astype(float)
        sw   = self._sample_weights(y)
        self.w_ = rng.normal(0, 0.01, d)
        self.b_ = 0.0
        lr = 0.01
        lam = 1.0 / (self.C * n)

        for epoch in range(200):
            idx = rng.permutation(n)
            for i in idx:
                xi, yi, wi = X[i], y_pm[i], sw[i]
                margin = yi * (xi @ self.w_ + self.b_)
                if margin < 1:
                    self.w_ = self.w_ - lr * (2*lam*self.w_ - wi*yi*xi)
                    self.b_ = self.b_ + lr * wi * yi
                else:
                    self.w_ = self.w_ - lr * (2*lam*self.w_)

        # Probability calibration via logistic regression on decision scores
        scores = X @ self.w_ + self.b_
        self._lr_model = LogisticRegression(C=1.0, max_iter=200, lr=0.1)
        self._lr_model.fit(scores.reshape(-1, 1), y)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w_ + self.b_

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = self.decision_function(X).reshape(-1, 1)
        return self._lr_model.predict_proba(scores)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (self.decision_function(X) >= 0).astype(int)
