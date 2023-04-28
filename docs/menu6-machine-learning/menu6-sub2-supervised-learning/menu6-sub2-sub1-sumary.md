---
layout: default
title: Summary
parent: Supervised Learning
grand_parent: Machine Leaning
nav_order: 1
---

# Summary
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
## STEP 1. Linear Regression

|item|expression|description|
|---|---|---|
|**hypothesis function**|<img src="./../../../images/menu6-sub2-sub2-linear-regression/image-20230415193845387.png" alt="image-20230415193845387" style="zoom: 80%;" />|prediction line using θ|
|**error**|![image-20230415211202619](./../../../images/menu6-sub2-sub2-linear-regression/image-20230415211202619.png)||
|**loss function**|<img src="./../../../images/menu6-sub2-sub2-linear-regression/image-20230415195035674.png" alt="image-20230415195035674" style="zoom:80%;" />|J(θ) mean for error sqare|
|**gradient descent**|<img src="./../../../images/menu6-sub2-sub2-linear-regression/image-20230415195202200.png" alt="image-20230415195202200" style="zoom:80%;" />|θ update by partial derivative J(θ)<br>- J(θ)를 각 θ에 대하여 편미분 후 부호를 반대로 바꾸면 가장 빨리 내려갈 수 있는 방향(loss function이 작은 값을 가질 수 있는)으로 이동<br>-가장 작은 J(θ)를 가지는 H(x)가 가장 성능이 좋음|