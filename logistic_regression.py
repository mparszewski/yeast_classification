import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load dataset
df = pd.read_csv("yeast.csv")

# take a look at the data
df.columns
df.dtypes
df.shape
df.describe()

# visaulisation
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
cls_lr = LogisticRegression(max_iter=150, C=7, solver='newton-cg', random_state=42)
cls_lr.fit(X_train, y_train)

# predict labels for training set
y_pred = cls_lr.predict(X_test)

# check model performance

# accuracy
print("Accuracy for logistic regression is equal to", np.round(accuracy_score(y_test, y_pred), 3))

# confusion matrix
labels = list(set(y))
cm = confusion_matrix(y_test, y_pred, labels)
df_cm = pd.DataFrame(cm, range(10), range(10))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 15}) # font size
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels, rotation = 45)
plt.title("Macierz pomyłek dla regresji logistycznej", size=14)
plt.show()


