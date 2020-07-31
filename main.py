from decision_tree import Decision_Tree
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree


# Test w/ Iris dataset using my class
dataset = load_iris()
X, y = dataset.data, dataset.target
clf_iris = Decision_Tree(max_depth = 5)
# Test to make target class strings instead of integers
y = ["one" if val == 1 or val == 2 else "zero" for val in y]
y = np.array(y)
# Need to ordinally encode strings to integers
if "int" not in str(y.dtype):
    # Reshape y array so it works w/ ordinal encoder
    y = y.reshape(-1, 1)
    encoder = OrdinalEncoder()
    y = encoder.fit_transform(y)
y = y.astype(int)
y = y.reshape(y.size,)

clf_iris.fit(X, y)
temp1 = np.array([[3, 2, 1, .5]])
temp2 = np.array([[4, 2.9, 1.3, .2]])
temp3 = np.array([[3.8, 3, 1.4, .4]])
temp4 = np.array([[7.7, 2.8, 6.7, 2]])


#temp1
print("------------------------------------------------------")
print(f"My Iris prediction for {temp1}:\n", clf_iris.predict(temp1))
print("------------------------------------------------------")
# Test w/ Iris dataset using sklearn
skl_clf_iris = DTC(splitter="best",random_state=42, max_depth=5)
skl_clf_iris.fit(X,y)
skl_preds_iris = skl_clf_iris.predict(temp1)
print(f"SKLearn Iris prediction for {temp1}:\n",skl_preds_iris)
print("------------------------------------------------------")


#temp2
print("------------------------------------------------------")
print(f"My Iris prediction for {temp2}:\n", clf_iris.predict(temp2))
print("------------------------------------------------------")
# Test w/ Iris dataset using sklearn
skl_clf_iris = DTC(splitter="best",random_state=42, max_depth=5)
skl_clf_iris.fit(X,y)
skl_preds_iris = skl_clf_iris.predict(temp2)
print(f"SKLearn Iris prediction for {temp2}:\n",skl_preds_iris)
print("------------------------------------------------------")


#temp3
print("------------------------------------------------------")
print(f"My Iris prediction for {temp3}:\n", clf_iris.predict(temp3))
print("------------------------------------------------------")
# Test w/ Iris dataset using sklearn
skl_clf_iris = DTC(splitter="best",random_state=42, max_depth=5)
skl_clf_iris.fit(X,y)
skl_preds_iris = skl_clf_iris.predict(temp3)
print(f"SKLearn Iris prediction for {temp3}:\n",skl_preds_iris)
print("------------------------------------------------------")


#temp4
print("------------------------------------------------------")
print(f"My Iris prediction for {temp4}:\n", clf_iris.predict(temp4))
print("------------------------------------------------------")
# Test w/ Iris dataset using sklearn
skl_clf_iris = DTC(splitter="best",random_state=42, max_depth=5)
skl_clf_iris.fit(X,y)
skl_preds_iris = skl_clf_iris.predict(temp4)
print(f"SKLearn Iris prediction for {temp4}:\n",skl_preds_iris)
print("------------------------------------------------------")



