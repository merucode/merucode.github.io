---
layout: default
title: Numpy Summary
parent: Numpy
grand_parent: Python
nav_order: 1
---

# Numpy Summary
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
## STEP 1. BASIC

```python
### Craet
array1 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
array1 = np.full(6, 7)          # [7 7 7 7 7 7]
array1 = np.zeros(6, dtype=int) # [0 0 0 0 0 0]
array1 = np.ones(6, dtype=int)  # [1 1 1 1 1 1]
array1 = np.random.random(6)    # [0.42214929 0.45275673 0.57978413 0.61417065 0.39448558 0.03347601]
array1 = numpy.arange(6)        # [0 1 2 3 4 5]
array1 = numpy.arange(2, 7)     # [2 3 4 5 6]
array1 = numpy.arange(3, 17, 3) # [3 6 9 12 15]

### Filter
array1 = np.array([3,4,5,5])
filter = np.where(array1 > 4) # array([2, 3])		
array1[filter]		          # array([5, 5])

### statistics
array1.max()    # 최댓값  
array1.min()    # 최솟값
array1.mean()   # 평균값
array1.std()    # 표준 편차  
array1.var()    #  분산

```
