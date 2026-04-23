# pip install imbalanced-learn scikit-learn

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN

# 1) Create imbalanced toy data (95% class 0, 5% class 1)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=1,
    weights=[0.95, 0.05],
    random_state=42
)

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Before ADASYN:", {0: (y_train==0).sum(), 1: (y_train==1).sum()})

# 3) Resample training data only
adasyn = ADASYN(random_state=42, n_neighbors=5)
X_train_bal, y_train_bal = adasyn.fit_resample(X_train, y_train)

print("After ADASYN: ", {0: (y_train_bal==0).sum(), 1: (y_train_bal==1).sum()})

# 4) Train model on resampled data
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_bal, y_train_bal)

# 5) Evaluate on untouched test set
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
