---
layout: default
title: Calculus Basic
parent: Calculus
grand_parent: Math
nav_order: 1
---

# Calculus
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
## STEP 1. Differential

### Step 1-1. basic

 * **differential** :  instantaneous rate of change for function

 * **Meaning of the slope in the graph**

  |positive slope|negative slope|
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415231348915.png" alt="image-20230415231348915" style="zoom:50%;" /> | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415231409020.png" alt="image-20230415231409020" style="zoom:50%;" /> |




* **Meaning of the slope(differential value) zero in the graph** 

  | 극소점 (Local Minimum)                                       | 극대점 (Local Maximum)                                       | 안장점 (Saddle Point)                                        |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415231523043.png" alt="image-20230415231523043" style="zoom:50%;" /> | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415231536925.png" alt="image-20230415231536925" style="zoom:50%;" /> | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415231556758.png" alt="image-20230415231556758" style="zoom:50%;" /> |

  

<br>

### Step 1-2. high order differential

* **partial derivative** : 편미분

* **Example**

  | function              | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415232114922.png" alt="image-20230415232114922" style="zoom:80%;" /> |
  | --------------------- | ------------------------------------------------------------ |
  | **x에 대해서 편미분** | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415232128995.png" alt="image-20230415232128995" style="zoom:80%;" /> |
  | **y에 대해서 편미분** | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415232147019.png" alt="image-20230415232147019" style="zoom:80%;" /> |
  | **합치기**            | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415232213160.png" alt="image-20230415232213160" style="zoom:80%;" /> |

  | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415232321051.png" alt="image-20230415232321051" style="zoom:65%;" /> | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415232337685.png" alt="image-20230415232337685" style="zoom:65%;" /> |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415232517251.png" alt="image-20230415232517251" style="zoom:65%;" /> | <img src="./../../../images/menu6-sub9-sub1-calculus/image-20230415232538608.png" alt="image-20230415232538608" style="zoom:67%;" /> |
  
* 편미분 개념과 기울기 개념은 인풋 변수가 아무리 많아도(고차원의 경우에도) 똑같이 적용 가능
* 편미분을 통해 기울기 벡터를 구할 수 있고, 이 기울기 벡터는 가장 가파르게 올라가는 방향을 가리킴<br>→ **기울기 벡터의 - 값은 가장 가파르게 내려가는 방향**

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. Composite Function And Chain Rule

### Step 2-1. Composite Function

*When f(y) = y<sup>3</sup>, y(x) = x<sup>2</sup> + 2x + 1*

*f(x) = f(y(x)) = (x<sup>2</sup> + 2x + 1)<sup>3*</sup>



### Step 2-2. Chain Rule

![image-20230425122742428](./../../../images/menu6-sub9-sub1-calculus/image-20230425122742428.png)

* Generalize

  ![image-20230425122802253](./../../../images/menu6-sub9-sub1-calculus/image-20230425122802253.png)
