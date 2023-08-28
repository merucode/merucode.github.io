---
layout: default
title: Data Type
parent: Python Basic
grand_parent: Python
nav_order: 2
---

# Data Type

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

## STEP 1. int



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 2. str



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 3. List

```python
### list comprehension
int_list = [1, 2, 3, 4, 5, 6]
squares = [x**2  for x in int_list]
print(squares) # [1, 4, 9, 16, 25, 36]
```


### Step 3-9. 중첩 리스트 합치기

```python
# 중첩 리스트 합치기
my_list = [[7, 6], [5, 4], [1, 2, 3], ['a', 'b']]

sum(my_list, [])    # [7, 6, 5, 4, 1, 2, 3, 'a', 'b']
sum(my_list)        # 오류 발생
```

* [https://blockdmask.tistory.com/558](https://blockdmask.tistory.com/558)



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 4. Dictionary

```python
### create dict 
basic_dict = dict{key1:value1, key2:value2} # 기본
list_dict = dict(zip(list1,list2)) # list 두개로 dict 만들기

### for statement using dict
for key, value in basic_dict.items():
for key in basic_dict.keys():
for value in basic_dict.values():



```


### Step 4-2. dict 병합

* [Python - 두개의 딕셔너리 병합 (merge)](https://codechacha.com/ko/python-merge-two-dict/#5-%EC%A4%91%EB%B3%B5%EB%90%9C-key%EC%9D%98-value%EB%A5%BC-%EB%AA%A8%EB%91%90-%EB%A6%AC%EC%8A%A4%ED%8A%B8%EC%97%90-%EC%A0%80%EC%9E%A5)

* [여러 dict에서 같은 key를 가진 value 더하여 병합하기](https://www.geeksforgeeks.org/python-sum-list-of-dictionaries-with-same-key/)

  ```python
  import collections, functools, operator
  
  # Initialising list of dictionary
  ini_dict_list = [{'a':5, 'b':10, 'c':90},
              {'a':45, 'b':78},
              {'a':90, 'c':10}]
  
  # printing initial dictionary
  print ("initial dictionary", str(ini_dict))
  
  # sum the values with same keys
  result = dict(functools.reduce(operator.add,
          map(collections.Counter, ini_dict_list)))
  
  print("resultant dictionary : ", str(result))

  # initial dictionary [{‘b’: 10, ‘a’: 5, ‘c’: 90}, {‘b’: 78, ‘a’: 45}, {‘a’: 90, ‘c’: 10}] 
  # resultant dictionary : {‘b’: 88, ‘a’: 140, ‘c’: 100}
  ```


<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 5. tuple



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 6. set







<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 7. bool

