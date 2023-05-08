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

  numbers.forEach((e, i) => console.log(e + numbers2[i]));
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

  const twiceNumbers2 = numbers.map((e) => e * 2); 
  ```

### Step 1-3. filter

* 배열의 요소를 하나씩 살펴보면서 콜백함수가 리턴하는 조건과 일치하는 요소만 모아서 **새로운 배열**을 리턴하는 메소드
* 요소로 반환 받으려면 `...`(spread) 사용

  ```javascript
  const devices = [
    {name: 'GalaxyNote', brand: 'Samsung'},
    {name: 'MacbookPro', brand: 'Apple'},
    {name: 'Gram', brand: 'LG'},
    {name: 'SurfacePro', brand: 'Microsoft'},
    {name: 'ZenBook', brand: 'Asus'},
    {name: 'MacbookAir', brand: 'Apple'},
  ];

  const apples = devices.filter((element, index, array) => {
    return element.brand === 'Apple';
  });

  console.log(apples); // (2) [{name: "MacbookPro", brand: "Apple"}, {name: "MacbookAir", brand: "Apple"}]

  console.log(...apples); // { name: 'MacbookPro', brand: 'Apple' } { name: 'MacbookAir', brand: 'Apple' }

  const apples2 = devices.filter((e) => e.brand === 'Apple' && e.name !== 'MacbookPro');

  console.log(...apples2); // { name: 'MacbookAir', brand: 'Apple' }
  ```

### Step 1-4. find

* `filter` 메소드와 비슷하게 동작하지만, 배열의 요소들을 반복하는 중에 콜백함수가 리턴하는 조건과 일치하는 **가장 첫번째 요소를 리턴하고 반복 종료**

  ```javascript
  const devices = [
    {name: 'GalaxyNote', brand: 'Samsung'},
    {name: 'MacbookPro', brand: 'Apple'},
    {name: 'Gram', brand: 'LG'},
    {name: 'SurfacePro', brand: 'Microsoft'},
    {name: 'ZenBook', brand: 'Asus'},
    {name: 'MacbookAir', brand: 'Apple'},
  ];

  const myLaptop = devices.find((element, index, array) => {
    console.log(index); // 콘솔에는 0, 1, 2까지만 출력됨.
    return element.name === 'Gram';
  });

  const myLaptop2 = devices.find((e) => e.name === 'Gram');

  console.log(myLaptop2); // { name: 'Gram', brand: 'LG' }
  ```

### Step 1-7. sort

* 배열을 정렬
* 메소드를 실행하는 원본 배열의 요소들을 정렬하기 때문에 필요 시 원본 복사 후 수행

  ```javascript
  const numbers = [1, 10, 4, 21, 36000];

  // 오름차순 정렬
  numbers.sort((a, b) => a - b);
  console.log(numbers); // (5) [1, 4, 10, 21, 36000]

  // 내림차순 정렬
  numbers.sort((a, b) => b - a);
  console.log(numbers); // (5) [36000, 21, 10, 4, 1]
  ```

### Step 1-8. reverse

* 배열의 순서를 뒤집어 주는 메소드
* 원본 배열의 요소들을 뒤집기 때문에 필요 시 원본 복사 후 수행

  ```javascript
  const letters = ['a', 'c', 'b'];
  const numbers = [421, 721, 353];

  letters.reverse();
  numbers.reverse();

  console.log(letters); // (3) ["b", "c", "a"]
  console.log(numbers); // (3) [353, 721, 421]
  ```



<br>


## STEP 2. 웹 개발 기초

* 크롬 개발자도구-console 에서 연습 가능

### Step 2-1. fetch

```javascript
fetch('https://www.google.com')
  .then((response) => response.text())
  .then((result) => { console.log(result); });

/* 
fetch('https://www.google.com')
  .then((response) => { console.log(response); }); 

위 코드로는 
response 파라미터로는 리스폰스의 실제 내용 자체가 넘어오는 게 아닙니다. response 파라미터에는, 리스폰스에 관한 각종 부가 정보들과, 실제 내용을 함께 갖고 있는 하나의 객체(object)가 넘어오기 때문에
내용을 보기 위해선
reponse 객체의 text라는 메소드를 호출해야 합니다. 그리고 이 text 메소드의 리턴값이 바로 리스폰스의 실제 내용입니다.
*/
```

위 코드의 fetch 함수의 실행 원리를 다시 정리하자면,

1. fetch 함수는 어떤 객체를 리턴하는데(Promise 객체, 챕터 3에서 배웁니다)
2. 이 객체의 then 메소드로, '리스폰스가 왔을 때 실행할 콜백'을 등록할 수 있습니다.
3. 이렇게 등록된 콜백들은 then 메소드로 등록한 순서대로 실행되고, 이때 이전 콜백의 리턴값을 이후 콜백이 넘겨받아서 사용할 수 있음


### step 2-2. 