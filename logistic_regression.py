import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# xgboost
# read in data

X = df.iloc[:, 1:9].values
y = df.iloc[:, 9].values

x_boost_train, x_boost_test, y_boost_train, y_boost_test = train_test_split(X, y, test_size=0.20, random_state=42)

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_boost_train)
label_encoder = label_encoder.fit(y_boost_test)
label_encoded_y_train = label_encoder.transform(y_boost_train)
label_encoded_y_test = label_encoder.transform(y_boost_test)

dtrain = xgb.DMatrix(x_boost_train, label_encoded_y_train)
dtest = xgb.DMatrix(x_boost_test, label_encoded_y_test)
# specify parameters via map
param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax', 'num_class': 8}
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)

