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

### Step 2-1. [hypothesis function](https://www.codeit.kr/learn/3990)

### Step 2-2. [model visualized](https://www.codeit.kr/learn/3990)

### Step 2-3. Predict

* [prediction 01:45](https://www.codeit.kr/learn/3991)
* [prediction 03:51](https://www.codeit.kr/learn/3991)


<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. Ann Basic

### Step 3-1. Hypothesis Function
 
 * [image] https://www.codeit.kr/learn/3995

### Step 3-2. Model Visualized(Layer)

 * [image] https://www.codeit.kr/learn/3995
 
### Step 3-3. Predict

 * [image] https://www.codeit.kr/learn/3995
 
### Step 3-4. Expression

* **parameter**

|Items|Description|Note|
|---|---|---|
|Layer|L = Number of hidden layer + 1||
|Output|[image](https://www.codeit.kr/learn/3999)|l: layer<br>i:th|
|Weight||reverse of matrix expression<br>i: l-1 th, j: l th|
|Bias|||

* **input/output data**

|input|output|
|---|---|
|[image](https://www.codeit.kr/learn/4003)||
|||


<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 4. Forward Propagation

### Step 4-1. Output Calculation

[image](https://www.codeit.kr/learn/4006)

### Step 4-2. Layer Calculation

### Step 4-3. Forward propagation

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 5. Hypothesis and Loss Function

### Step 5-1. Hypothesis Function

[image](https://www.codeit.kr/learn/4009)

### Step 5-2. Loss Function

[loss function](https://www.codeit.kr/learn/4009)
[ex](https://www.codeit.kr/learn/4009)
* *h<sub>w</sub>(x<sup>(i)</sup>) = a<sub>i</sub><sup>[L]</sup>*

###  Step 5-3. Non Convex function

* [graph](https://www.codeit.kr/learn/4013)
* [reason](https://www.codeit.kr/learn/4013)

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 6. Back Propagation

* Reference : [Math] - [Composite Function And Chain Rule]

### Step 6-1. Partial Derivative