---
layout: default
title: ANN Basic
parent: Supervised Learning
grand_parent: Deep Leaning
nav_order: 6
---

# ANN Basic
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
## STEP 1. MNIST Dataset

* MNIST Dataset  : Black and white handwritten digits dataset
	* 28x28 → 784 pixel data
	* Express the gray scale as a decimal between 0 and 1, 0~1 is result of min-max normalization(0~255)
	* Expression of MNIST data 5
		`([0, 0, 0, 0, ..., 0.66, 0.12, 0.99, 0.80, 0.77, 0.55, ..., 0, 0, 0, 0], 5)`

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. Review Rogistic Regression

### Step 2-1. Hypothesis Function

![image-20230425122922521](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425122922521.png)

### Step 2-2. Model Cisualized

![image-20230425122946772](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425122946772.png)

### Step 2-3. Predict

| Predict                                                      | Predict                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20230425123042177](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425123042177.png) | ![image-20230425123055739](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425123055739.png) |



<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. Ann Basic

### Step 3-1. Hypothesis Function

* Purpose of a logistic regression model is to find the parameter theta values that best fit the data, the purpose of an artificial neural network is to find the weights and biases that best fit the given data
* From logistic regression, θ<sub>0</sub> is marked as b, the rest of θ is marked as w

![image-20230425123159684](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425123159684.png)

### Step 3-2. Model Visualized(Layer)

| Model                                                        | Learning                                                     | Activation                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20230425123236957](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425123236957.png) | <img src="./../../../images/menu6-sub4-sub6-ann-basic/image-20230425123952933.png" alt="image-20230425123952933" style="zoom:67%;" /> | <img src="./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124309205.png" alt="image-20230425124309205" style="zoom:67%;" /> |



### Step 3-3. Expression

* **parameter**

|Items|Description|Note|
|---|---|---|
|Layer|L = Number of hidden layer + 1||
|Output|![image-20230425124426814](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124426814.png)|l: layer<br>i:th|
|Weight|![image-20230425124441252](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124441252.png)|**reverse of matrix expression**<br>i: l-1 th, j: l th|
|Bias|![image-20230425124459679](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124459679.png)||

* **input/output data**

|Input|Output(One-hot encoding)|
|---|---|
|![image-20230425124709395](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124709395.png)|![image-20230425124718756](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124718756.png)|
|![image-20230425124626202](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124626202.png)|![image-20230425124638002](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124638002.png)|


<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 4. Forward Propagation

### Step 4-1. Output Calculation

<img src="./../../../images/menu6-sub4-sub6-ann-basic/image-20230425125145254.png" alt="image-20230425125145254" style="zoom: 80%;" />

### Step 4-2. Layer Calculation

| Component                                                    | Calculation                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20230425124904128](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124904128.png) | ![image-20230425124913687](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425124913687.png) |



### Step 4-3. Forward propagation

![image-20230425125045722](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425125045722.png)

<br>	

<!------------------------------------ STEP ------------------------------------>
## STEP 5. Hypothesis and Loss Function

### Step 5-1. Hypothesis Function

* Hypothesis function that calculates the **output of the last layer neurons** according to the given weights and biases.\

  | Hypothesis function                                          | Ex                                                           |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![image-20230425125333070](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425125333070.png) | ![image-20230425125415331](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425125415331.png) |



### Step 5-2. Loss Function(MSE)

| Loss function(MSE)                                           | Ex                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20230425125654296](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425125654296.png) | ![image-20230425125810458](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425125810458.png)![image-20230425125821238](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425125821238.png) |

* *h<sub>w</sub>(x<sup>(i)</sup>) = a<sub>i</sub><sup>[L]</sup>*

###  Step 5-3. Non Convex function(MSE loss function in ANN)

| Ann MSE Loss function Non Convex function                    | Reason to use graedent descent                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./../../../images/menu6-sub4-sub6-ann-basic/image-20230425130030771.png" alt="image-20230425130030771" style="zoom:80%;" /> | ![image-20230425130051332](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425130051332.png) |



<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 6. Back Propagation

* [math reference](https://merucode.github.io/docs/menu6-deep-learning/menu6-sub9-math/menu6-sub9-sub1-calculus.html#step-2-composite-function-and-chain-rule)

* Purpose of back progagation : **Update weight and bias(gradient descent)**
  * Purpose of forward progation : **Calculate hypothesis function output **

### Step 6-1. Partial Derivative

![image-20230425133922123](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133922123.png)

| Item               | Expression                                                   | Image                                                        | Note           |
| :----------------- | :----------------------------------------------------------- | ------------------------------------------------------------ | -------------- |
| **Component**      | ![image-20230425132917017](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425132917017.png) |                                                              |                |
| **Weight**         | ![image-20230425133027983](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133027983.png)<br>![image-20230425133119317](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133119317.png) | ![image-20230425133718767](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133718767.png) | Update W       |
| **Bias**           | ![image-20230425133351428](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133351428.png)<br>![image-20230425133405567](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133405567.png) | ![image-20230425133726545](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133726545.png) | Update b       |
| **Pre activation** | ![image-20230425133505318](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133505318.png)<br>![image-20230425133438194](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133438194.png) | ![image-20230425133742813](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425133742813.png) | Calculate W, b |



### Step 6-2. Chain Rule

| Before apply to chain rule                                   | After apply to chain rule                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./../../../images/menu6-sub4-sub6-ann-basic/image-20230425134023617.png" alt="image-20230425134023617" style="zoom: 80%;" /> | <img src="./../../../images/menu6-sub4-sub6-ann-basic/image-20230425134107852.png" alt="image-20230425134107852" style="zoom:80%;" /> |



### Step 6-3. Gradient Descent

![image-20230425134358633](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425134358633.png)



### Step 6-4. Matrix Expression

![image-20230425140016234](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425140016234.png)

| Items              | Expression                                                   | Maxtix                                                       | Maxtix Expression                                            |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Simplification** | ![image-20230425135139397](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135139397.png)<br>![image-20230425135203082](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135203082.png)<br>![image-20230425135215292](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135215292.png) |                                                              |                                                              |
| **Bias**           | ![image-20230425135247293](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135247293.png) | ![image-20230425135306142](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135306142.png) | ![image-20230425135320760](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135320760.png) |
| **Weight**         | ![image-20230425135421569](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135421569.png)<br>![image-20230425135440956](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135440956.png)<br>![image-20230425135510768](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135510768.png)<br>![image-20230425135527656](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135527656.png) | ![image-20230425135617583](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135617583.png) | ![image-20230425135635560](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135635560.png) |
| **Pre activation** | ![image-20230425135828356](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135828356.png)<br>![image-20230425135840961](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135840961.png)<br>![image-20230425135908946](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135908946.png) | ![image-20230425135921143](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425135921143.png) | ![image-20230425140007008](./../../../images/menu6-sub4-sub6-ann-basic/image-20230425140007008.png) |

