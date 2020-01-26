import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv("yeast.csv")

# take a look at the data
df.columns
df.dtypes
df.shape

# split to x and y
X = df.iloc[:, 1:9].values
y = df.iloc[:, 9].values

# split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# perform logistic regression and fit the model
cls_lr = LogisticRegression(random_state=42, max_iter=150, C=0.7, solver='newton-cg')
cls_lr.fit(X_train, y_train)

# predict labels for training set
y_pred = cls_lr.predict(X_test)

# check model performance
print("Accuracy for logistic regression is equal to", np.round(accuracy_score(y_test, y_pred), 3))

