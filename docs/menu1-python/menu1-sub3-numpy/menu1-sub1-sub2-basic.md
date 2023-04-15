---
layout: default
title: Numpy Basic
parent: Numpy
grand_parent: Python
nav_order: 2
---

# Numpy Basic
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
## STEP 1. numpy array 생성방법
### Step 1-1. 방법
* **`np.array(list)`** : python list에서 생성
* **`np.full(counts, value)`** : 균일한 값 생성
* **`np.zeros(counts, dtype=int)`** : 0 생성
* **`np.ones(counts, dtype=int)`** : 1 생성 
* **`np.random.random(counts)`** :  random 생성
* **`np.arange()`** : 연속된 값 생성

### Step 1-2. 예제
```python
array1 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
# [ 2  3  5  7 11 13 17 19 23 29 31]

array1 = np.full(6, 7)
# [7 7 7 7 7 7]

array1 = np.zeros(6, dtype=int)
# [0 0 0 0 0 0]

array1 = np.ones(6, dtype=int)
# [1 1 1 1 1 1]

array1 = np.random.random(6)
# [0.42214929 0.45275673 0.57978413 0.61417065 0.39448558 0.03347601]

array1 = numpy.arange(6)
# [0 1 2 3 4 5]
array1 = numpy.arange(2, 7)
# [2 3 4 5 6]
array1 = numpy.arange(3, 17, 3)
# [3 6 9 12 15]
```
  
<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. numpy boolean 연산
### Step 2-1. 방법
```python
array1 = np.array([3,4,5])
array1 > 4
# array([False, False, True])
array1 % 4 == 0
# array([False, True, False])
```

### Step 2-2. 사용법
```python
array1 = np.array([3,4,5,5])
filter = np.where(array1 > 4)
# array([2, 3])		# array1 > 4 에서 Ture인 요소의 index만 추출
array1[filter]		# [] 대괄호 사용
# array([5, 5])		# array1에서 4보다 큰 요소들만 추출
```
  
<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. numpy 기본 통계 연산
```python
import numpy as np 
array1 = np.array([14, 6, 13, 21, 23, 31, 9, 5])

print(array1.max()) # 최댓값  
print(array1.min()) # 최솟값
print(array1.mean()) # 평균값
print(array1.std()) # 표준 편차  
print(array1.var()) # 분산

# 중앙값
array1 = np.array([8, 12, 9, 15, 16]) 
array2 = np.array([14, 6, 13, 21, 23, 31, 9, 5])
print(np.median(array1)) # 중앙값 12.0 
print(np.median(array2)) # 중앙값 13.5(중앙값이 13과 14 두 개 평균값 13.5)
```

<br>
