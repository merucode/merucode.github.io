---
layout: default
title: Decision Trees and Ensemble
parent: Supervised Learning
grand_parent: Machine Leaning
nav_order: 15
---

# Decision Trees and Ensemble
{: .no_toc .d-inline-block }
ing
{: .label .label-green }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

<!------------------------------------ STEP ------------------------------------>
## STEP 1. Decision Tree

### Step 1-1. Decision Tree

![image-20230426121052440](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121052440.png)

* In machine learning, **node question is made by learning**
	* **Purpose of decision tree is we find the nodes** that can best classify the data though classifying the training data, 
	
### Step 1-2. Gini Impurity

* **Gini impurity** : The degree of shuffling of the data set
	* High value mean more impure

|Example Function|Example Data|
|---|---|
|![image-20230426121258032](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121258032.png)|![image-20230426121219105](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121219105.png)<br>![image-20230426121244413](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121244413.png)|

### Step 1-3. Node

* **Good Node** : **Low gini impourity** of dataset divided by node 
	* Node Evaluation Example
		![image-20230426121415074](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121415074.png)
* Type of Node : Classification, Question
* Node Selection
	![image-20230426121450989](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121450989.png)
* Depth of decision tree
![image-20230426121525083](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121525083.png)
* Node for numerical features
![image-20230426121613151](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121613151.png)

### Step 1-4. Feature Importance


|Node importance|Example|
|---|---|
|![image-20230426121745528](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121745528.png)|![image-20230426121706479](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121706479.png)|
|**Feature Importance**|**Feature Importance**|
|![image-20230426121801087](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121801087.png)|![image-20230426121810949](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121810949.png)|


### Step 1-5. sklearn

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

cancer_data = load_breast_cancer()

X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()    # not occur error massage for learning

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print(predictions, score)

'''중요도 시각화
importances = model.feature_importances_ 		
indices_sorted = np.argsort(importances)

plt.figure()
plt.title("Feature importance")
plt.bar(range(len(importances)), importancse[indices_sorted])
plt.xticks(range(len(importances)), X.columns[indeics_sorted], rotation=90)
'''
```

<br>

## STEP 2. Random Forest

### Step 2-1. Ensemble

* Decision trees disadvantage :  **inaccuracy(low performance)**
	* Why learn decision trees?  By application, other models with good performance can be created
* **Ensemble** : Building a number of models and combining their predictions to create a comprehensive forecast

### Step 2-2. Random Forest(bagging)

* Bootstrapping dataset
	![image-20230426121914216](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426121914216.png)
	
* Bagging

  ![image-20230426122000412](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122000412.png)

* Randomly Creating Decision Trees
	![image-20230426122050389](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122050389.png)
### Step 2-3. sklearn

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

cancer_data = load_breast_cancer()

X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = y_train.values.ravel()    # not occur error massage for learning

model = RandomForestClassifier(n_estimators=10, max_depth=4) # n_estimators: the number of random tree models
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print(predictions, score)
```

<br>

## STEP 3. Adaboost

### Step 3-1. Boosting

![image-20230426122129693](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122129693.png)

### Step 3-2. Adaboost

|Stump|Dataset|
|---|---|
|![image-20230426122159237](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122159237.png)|![image-20230426122232788](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122232788.png)|
|**Predict**|**Summary**|
|![image-20230426122246136](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122246136.png)|![image-20230426122257863](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122257863.png)|

### Step 3-3. Stump Performance

|Dataset|First Stump|
|---|---|
|<img src="./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122422420.png" alt="image-20230426122422420" style="zoom:80%;" />|![image-20230426122532765](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122532765.png)|
|**Performance**|**Summary**|
|<img src="./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122456776.png" alt="image-20230426122456776" style="zoom:80%;" />|![image-20230426122542621](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122542621.png)|

### Step 3-4. Update importance

|Weight function| graph                                                        |
|---|---|
|![image-20230426122755767](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122755767.png)|![image-20230426122806324](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122806324.png)|
|**Update**|**Rebalancing**|
|![image-20230426122622277](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122622277.png)|![image-20230426122628143](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426122628143.png)|

### Step 3-5. Update Stump

|Method|New Dataset|
|---|---|
|![image-20230426123434869](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426123434869.png)|![image-20230426123450005](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426123450005.png)|
|**Stump**|**Summary**|
|<img src="./../../../images/menu6-sub2-sub15-decision-trees/image-20230426123507084.png" alt="image-20230426123507084" style="zoom:80%;" />|<img src="./../../../images/menu6-sub2-sub15-decision-trees/image-20230426123519230.png" alt="image-20230426123519230" style="zoom:80%;" />|

### Step 3-6. Predict

![image-20230426123557599](./../../../images/menu6-sub2-sub15-decision-trees/image-20230426123557599.png)

### Step 3-7. skleran

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd

cancer_data = load_breast_cancer()

X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()    # not occur error massage for learning

model = AdaBoostClassifier(n_estimators=50, random_state=5)	# n_estimators: the number of stumps
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)

print(predictions, score)
```
