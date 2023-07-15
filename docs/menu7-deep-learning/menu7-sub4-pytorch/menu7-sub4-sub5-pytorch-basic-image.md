---
layout: default
title: Pytorch Basic Image
parent: Pytorch
grand_parent: Deep Learning
nav_order: 5
---

# Pytorch Basic Image
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

### 텐서 파일 이미지 확인 하기

```python
from torchvision.transforms import Compose, ToPILImage
import matplotlib.pyplot as plt

trans = Compose([ToPILImage()])

plt.imshow(trans(test_set[i][0]))
plt.show()
```














## STEP 1. Basic Coding Style

```python
```

<!------------------------------------ STEP ------------------------------------>

## STEP 2. Data Preprocessing

```python
```


<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 3. Module

* [[Blog] Conv2d 알아보기](https://blog.joonas.io/196)
* [[Blog] Convolution 연산에서 Filter에 대한 이해](https://woochan-autobiography.tistory.com/883)

<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 4. Learning



<br>


<!------------------------------------ STEP ------------------------------------>

## STEP 5. Evaluation



<br>