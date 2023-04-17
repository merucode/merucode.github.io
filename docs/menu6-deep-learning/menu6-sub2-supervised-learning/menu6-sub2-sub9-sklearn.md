---
layout: default
title: sklearn
parent: Supervised Learning
grand_parent: Deep Leaning
nav_order: 9
---

# sklearn
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
## STEP 1. Polynominal Regression

```python
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd  

diabetes_dataset = datasets.load_diabetes()

polynominal_transformer = PolynomialFeatures(2)
polynominal_data = polynominal_transformer.fit_transform(diabetes_dataset.data)
polynominal_feature_names = polynominal_transformer.get_feature_names(diabetes_dataset.feature_names)

X = pd.DataFrame(polynominal_data, columns=polynominal_feature_names)
y = pd.DataFrame(diabetes_dataset.target, columns=['diabetes'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = LinearRegression()
model.fit(x_train, y_train)

y_test_predict = model.predict(x_test)

rmse = mean_squared_error(y_test, y_test_predict) ** 0.5
```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. Logistic Regression

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import pandas as pd

iris_data = load_iris()
# iris_data.DESCR

x = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y= pd.DataFrame(iris_data.target, columns=['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

y_train = y_train.values.ravel() # 안써도 되나 경고 발생(경고 제거용?)

model = LogisticRegression(solver='saga', max_iter=2000)
# solver : 최적화 시 어떤 알고리즘 쓸지 결정
# max_iter : interation(possiple to stop on optimize before max_iter)
model.fit(x_train, y_train)

model.predict(x_test)
# logistic evaluation
model.score(x_test, y_test)
```
