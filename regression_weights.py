import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn import linear_model

# load data
dt = np.loadtxt("classification_stats.csv", delimiter=" ", dtype=float)
np.random.seed(2)
idx = np.random.choice(dt.shape[0], size=dt.shape[0], replace=False)
# shuffle data
dt = dt[idx, :]
# separate features and labels
X = dt[:,:-1]
y = dt[:,-1].astype(np.int)
# separate training and test data
train_cutoff = int(X.shape[0]*0.75)
X_train = X[:train_cutoff,:]
X_test = X[train_cutoff:,:]
y_train = y[:train_cutoff]
y_test = y[train_cutoff:]

# Train logistic regression on training data
clf = LogisticRegression(random_state=1).fit(X_train, y_train)
p = clf.predict_proba(X_test)
# Training accuracy
print(clf.score(X_train,y_train))
# Test accuracy
print(clf.score(X_test,y_test))
# Test cross-entropy
print(log_loss(y_test, p))

# Train logistic regression on full data
clf = LogisticRegression(random_state=1).fit(X, y)
predict=clf.predict(X)
fp = np.sum(predict[y==0.0])
fn = np.sum(y[predict==0.0])
p = clf.predict_proba(X)
# Training accuracy
print(clf.score(X,y))
# Training cross entropy
print(log_loss(y, p[:,1]))

# Parameters
print(clf.coef_)
print(clf.intercept_)


