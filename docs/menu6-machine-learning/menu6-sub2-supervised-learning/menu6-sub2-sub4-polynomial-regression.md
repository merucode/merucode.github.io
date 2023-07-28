---
layout: default
title: Polynomial Regression
parent: Supervised Learning
grand_parent: Machine Leaning
nav_order: 4
---

# Polynomial Regression
{: .no_toc }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

<!------------------------------------ STEP ------------------------------------>
## STEP 1. Polynomial Regression hypothesis function

| h<sub>θ</sub>(x)=θ<sub>0</sub>+θ<sub>1</sub>x(Linear)        | h<sub>θ</sub>(x)=θ<sub>0</sub>+θ<sub>1</sub>x+θ<sub>2</sub>x<sup>2</sup> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232310726.png" alt="image-20230416232310726" style="zoom:50%;" /> | <img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232323677.png" alt="image-20230416232323677" style="zoom:50%;" /> |
| **h<sub>θ</sub>(x)=θ<sub>0</sub>+θ<sub>1</sub>x+θ<sub>2</sub>x<sup>2</sup>+θ<sub>3</sub>x<sup>3</sup>** | **h<sub>θ</sub>(x)=θ<sub>0</sub>+θ<sub>1</sub>x+θ<sub>2</sub>x<sup>2</sup>+θ<sub>3</sub>x<sup>3</sup>+θ<sub>4</sub>x<sup>4</sup>** |
| <img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232339839.png" alt="image-20230416232339839" style="zoom:50%;" /> | <img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232351564.png" alt="image-20230416232351564" style="zoom:50%;" /> |

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. Single Feature

* Single feature polynomial regression is same as multiple linear regression
	
	<img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232510005.png" alt="image-20230416232510005" style="zoom: 50%;" />

|first order|Add more polynomial as like multiple LR|
|---|---|
|<img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232550496.png" alt="image-20230416232550496" style="zoom:50%;" />| <img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232612657.png" alt="image-20230416232612657" style="zoom:50%;" /> |

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. multiple polynomial function

* **multiple polynomial function** is same as **single feature polynomial function**
	* it represent **relation between features**
		* *ex> square width and height(meaning   
		area)*
	
* **Example**

	<img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232930854.png" alt="image-20230416232930854" style="zoom:80%;" />
	
	* **assume hypothesis function is quadratic term**
	
	  <img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232827069.png" alt="image-20230416232827069" style="zoom: 80%;" />
	  <img src="./../../../images/menu6-sub2-sub4-polynomial-regression/image-20230416232904678.png" alt="image-20230416232904678" style="zoom: 67%;" />

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 4. sklearn

```python
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures ### 다항속성 추가
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd

boston_dataset = load_boston()

polynominal_transformer = PolynomialFeatures(2) ### 가상함수 2차함수 가정
polynominal_data = polynominal_transformer.fit_transform(boston_dataset.data)
polynominal_feature_names = polynominal_transformer.get_feature_names(boston_dataset.feature_names)

polynominal_data.shape	# (506, 105) original data is (506, 13) 

x = pd.DataFrame(polynomial_data, columns=polynominal_feature_names)
y= pd.DataFrame(boston_dataset.target, columns=['MEDV'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, randome_state=5)

model = LinearRegression()
model.fit(x_train, y_train)

model.coef_
model.intercept_

y_test_prediction = model.predict(x_test)
mean_squared_error(y_test, y_test_prediction) ** 0.5
```