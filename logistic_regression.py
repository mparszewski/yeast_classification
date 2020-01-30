# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import xgboost as xgb

# load dataset
df = pd.read_csv("yeast.csv")

# take a look at the data
df.columns
df.dtypes
df.shape
df.describe()

# visualisation - logistic regression
fig1 = plt.figure()
df.class_distribution.value_counts().plot(kind="bar", title="Liczebności poszczególnych klas", color="green")
plt.show()
corr = df.corr()
fig2 = plt.figure()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
plt.title("Korelacja pomiędzy zmiennymi objaśniającymi")
plt.show()

# split to x and y
X = df.iloc[:, 2:9].values
y = df.iloc[:, 9].values

# split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# perform logistic regression and fit the model
cls_lr = LogisticRegression(max_iter=150,
                            C=7,
                            solver='newton-cg',
                            random_state=42)
cls_lr.fit(X_train, y_train)

# predict labels for training set
y_pred = cls_lr.predict(X_test)

# check model performance - logistic regression
# accuracy
print("Accuracy for logistic regression is equal to", np.round(accuracy_score(y_test, y_pred), 3))

# confusion matrix
labels = list(set(y))
cm = confusion_matrix(y_test, y_pred, labels)
df_cm = pd.DataFrame(cm, range(10), range(10))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 15})  # font size
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels, rotation=45)
plt.title("Macierz pomyłek dla regresji logistycznej", size=14)
plt.show()

# XGBOOST
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
label_encoded_y_train = label_encoder.transform(y_train)
label_encoded_y_test = label_encoder.transform(y_test)

class_weights = list(class_weight.compute_class_weight('balanced',
                                                       np.unique(df['class_distribution']),
                                                       df['class_distribution']))
w_array = np.ones(label_encoded_y_train.shape[0], dtype='float')

for i, val in enumerate(label_encoded_y_train):
    w_array[i] = class_weights[val - 1]

clf = xgb.XGBClassifier(colsample_bylevel=0.9,
                        colsample_bytree=0.9,
                        n_estimators=1000,
                        max_depth=20,
                        learning_rate=0.01,
                        class_weights=w_array,
                        min_child_weight=3,
                        gamma=3.0,
                        subsample=0.9,
                        reg_lambda=5.0,
                        random_state=42)

clf.fit(X_train, label_encoded_y_train)
y_pred_xgb = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# check model performance - logistic regression
# accuracy
ac_test = accuracy_score(label_encoded_y_test, y_pred_xgb)
ac_train = accuracy_score(label_encoded_y_train, y_pred_train)
print('Accuracy for xgboost on test data is equal to ', ac_test)
print('Accuracy for xgboost on train data is equal to ', ac_train)

# confusion matrix
cm = confusion_matrix(label_encoded_y_test, y_pred_xgb, list(range(10)))
df_cm = pd.DataFrame(cm, range(10), range(10))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 15})  # font size
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_, rotation=45)
plt.title("Macierz pomyłek dla XGBOOST", size=14)
plt.show()


# hyper parameters tuning with cross validation
# param_grid = {
#         'max_depth': [4, 6, 8, 10, 13, 15, 20],
#         'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
#         'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#         'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#         'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#         'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0],
#         'gamma': [0, 1.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
#         'reg_lambda': [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
#         'n_estimators': [100, 300, 500, 1000],
#         'class_weights': [w_array],
#         'random_state': [42]}
#
# rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=5000,
#                             n_jobs=1, verbose=2, cv=3,
#                             scoring='accuracy', refit=False, random_state=42)
#
# rs_clf.fit(X_train, y_train)
# best_score = rs_clf.best_score_
# best_params = rs_clf.best_params_
# print("Best score: {}".format(best_score))
# print("Best params: ")
# for param_name in sorted(best_params.keys()):
#     print('%s: %r' % (param_name, best_params[param_name]))
