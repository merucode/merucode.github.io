---
layout: default
title: Javascript Note
parent: Javascript
grand_parent: Frontend
nav_order: 1
---

# Javascript Note
{: .no_toc}

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
<!------------------------------------ STEP ------------------------------------>


## STEP 1. 배열 메소드

* [url](https://www.codeit.kr/learn/4530)

### Step 1-1. forEach

* forEach 메소드는 첫 번째 아규먼트로 콜백 함수를 전달받는데요. 콜백 함수의 파라미터에는 각각 `배열의 요소`, `index`, `메소드를 호출한 배열`이 전달됩니다.
* index와 array는 생략가능
* index를 이용하여 다른 배열과 동시 연산 가능

```javascript
const numbers = [1, 2, 3];
const numbers2 = [4, 5, 6];

numbers.forEach((element, index, array) => {
  console.log(element); // 순서대로 콘솔에 1, 2, 3 출력
});

numbers.forEach((element, i) => {
  console.log(element + numbers2[i]); // 순서대로 콘솔에 5, 7, 9 출력
});
```


### Step 1-2. map

* `forEach`와 유사하나, 첫 번째 아규먼트로 전달하는 콜백 함수가 매번 리턴하는 값들을 모아서 새로운 배열을 만들어 리턴
* index와 array는 생략가능

```javascript
const numbers = [1, 2, 3];
const twiceNumbers = numbers.map((element, index, array) => {
  return element * 2;
});

console.log(twiceNumbers); // (3) [2, 4, 6]
```

### Step 1-3. map

```javascript
```

```javascript
```

```javascript
```

```javascript
```
