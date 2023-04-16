---
layout: default
title: Multiple Linear Regression
parent: Supervised Learning
grand_parent: Deep Leaning
nav_order: 3
---

# Multiple Linear Regression
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
## STEP 1. Multiple Linear Regression

* **Multiple Linear Regression**
	* hard to visualized
	* principle is same to liear regression
* **Data Expression**
	* m is data number, n is feature(input) number
	* x<sub>j</sub><sup>(i)</sup> : (i) data, j feature

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. Hypothesis function

* **Hypothesis function for n features**
	* *h<sub>θ</sub>(x)  = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub> + ... + θ<sub>n</sub>x<sub>n</sub>*
	* *h<sub>θ</sub>(x)  =  θ<sub>T</sub>x*
		* *θ<sub>T</sub> = [θ<sub>0</sub>, θ<sub>1</sub>, θ<sub>2</sub>, ... , θ<sub>n</sub>]*
		* x = [1, x<sub>1</sub>, x<sub>2</sub>, ... , x<sub>n</sub>]

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. Gradient Descent

* **Loss Functions**
$J(θ) = \frac{1}{2m}\displaystyle\sum_{i=1}^{m}{(h_θ(x^{(i)})-y^{(i)})^2}$
*(θ = θ<sub>0</sub>, θ<sub>1</sub>, θ<sub>2</sub>, ... , θ<sub>n</sub>)*

* **θ Update** (result of partial derivative J(θ) by θ)
$θ_j = θ_j - α\frac{1}{m}\displaystyle\sum_{i=1}^{m}{(h_θ(x^{(i)})-y^{(i)})}x^{(i)}$
	*(j = 0, 1, 2, ..., n)*
	
* **Matrics Express**
	* $X = \begin{bmatrix} x_0^{(1)} & x_1^{(1)}\cdots & x_n^{(1)} \\ \vdots & \ddots & \vdots \\ x_0^{(m)} & x_1^{(m)} \cdots & x_n^{(m)} \end{bmatrix}$
	
	* $θ = \begin{bmatrix} θ_0 \\ θ_1 \\ \vdots \\ θ_n \end{bmatrix}$
	* $Xθ  = \begin{bmatrix} h_θ(x^{(1)}) \\ h_θ(x^{(2)}) \\ \vdots \\ h_θ(x^{(m)}) \end{bmatrix}$ 
	* $error = Xθ - y =  \begin{bmatrix} h_θ(x^{(1)})-y^{(1)} \\ h_θ(x^{(2)})-y^{(2)} \\ \vdots \\ h_θ(x^{(m)})-y^{(m)} \end{bmatrix}$
	* $X^T×error = \begin{bmatrix} x_0^{(1)} & x_0^{(2)}\cdots & x_0^{(m)} \\ \vdots & \ddots & \vdots \\ x_n^{(1)} & x_n^{(2)} \cdots & x_n^{(n)} \end{bmatrix} × \begin{bmatrix} h_θ(x^{(1)})-y^{(1)} \\ h_θ(x^{(2)})-y^{(2)} \\ \vdots \\ h_θ(x^{(m)})-y^{(m)} \end{bmatrix} = \begin{bmatrix}\displaystyle\sum_{i=1}^{m}{(h_θ(x^{(i)})-y^{(i)})}x_0^{(i)}\\\displaystyle\sum_{i=1}^{m}{(h_θ(x^{(i)})-y^{(i)})}x_1^{(i)}\\ \vdots \\\displaystyle\sum_{i=1}^{m}{(h_θ(x^{(i)})-y^{(i)})}x_n^{(i)}\end{bmatrix}$
		* **gradient descent for each feature**
	* **from** $θ_j = θ_j - α\frac{1}{m}\displaystyle\sum_{i=1}^{m}{(h_θ(x^{(i)})-y^{(i)})}x^{(i)}$(result of partial derivative J(θ) by θ)
	 $θ ← θ - α\frac{1}{m}(X^T × error)$