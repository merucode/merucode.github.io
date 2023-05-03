---
layout: default
title: React Basic
parent: React
grand_parent: Frontend
nav_order: 2
---

# React Basic
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

## STEP 1. React 시작하기

### Step 1-1. 개발환경 세팅하기

- node.js 설치

- **`bash`**

  ```bash
  # version 확인
  $ node -v
  $ npm -v  
  ```

### Step 1-2. 프로젝트 시작하기

- **`bash`**

  ```bash
  $ npm init react-app [폴더이름]
  $ npm init react-app .  # 현재 폴더에서 react 프로젝트 시작

  $ npm run start         # 개발서버 실행(localhost:3000)
  ```

### Step 1-3. React 개발자 도구

- 크롬 웹스토어 확장 프로그램 : react develop tools

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. React 개발 기초

### Step 2-1. 프로젝트 세팅

- **`bash`**

  ```bash
  npm init react-app .   # 현재 폴더에서 react 프로젝트 시작
  ```

- 불필요 파일 삭제
  - public 폴더에서 index.html 외 모두 제거
  - src 폴더에서 index.js 외 모두 제거

- **`index.html`**

  ```html
  <!DOCTYPE html>
  <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <title>주사위 게임</title>
    </head>
    <body>
      <div id="root"></div>
    </body>
  </html>
  ```


- **`index.js`**

  ```javascript
  import ReactDOM from 'react-dom/client';

  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(<h1>안녕 리액트!</h1>);
  ```
  - 수업에서는 ReactDOM.render() 함수 안에서 코드를 작성하는데, 최신 버전을 사용하시는 분들은 root.render() 함수 안에서 작성

### Step 2-2. JSX 문법

- react에서 제공하는 js와 html을 함께 사용가능한 문법
- index.js의 root.render 내에 JSX 문법 입력
- html과 차이(Camel Case)

|html|JSX|
|---|---|
|class|className|
|for|htmlFor|
|onmousedown<br>이벤트핸들러|onMouseDown|

### Step 2-3. Fragment

- JSX는 하나의 요소이여야 함
- `<div>`를 이용해서 하나로 묶을 수 있으나, `<div>`를 사용하기 않기를 원할 경우 `<Fragment>` 혹은 `<>` 사용 가능
- `<Fragment>` 사용 시 `import { Fragement } from 'react';` 필요(축약형인 `<>`은 import 불필요)
- 예제

  ```javascript
  ReactDOM.render(
    <>
      <p>안녕</p>
      <p>리액트!</p>
    </>,
    document.getElementById('root')
  );
  // <></> 없이 사용 시 오류 발생
  ```

### Step 2-4. JSX에서 javascript 표현식 사용하기

- 중괄호`{}` 사용 javascript 표현식 사용
- 중괄호 안에서 for, if문 등의 javascript 문장식 사용 불가
  - 필요하다면 조건 연산자, 배열의 반복 메소드를 활용

- 예제

  ```javascript
  import ReactDOM from 'react-dom';

  const product = 'MacBook';
  const model = 'Air';
  const imageUrl =
    'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/MacBook_with_Retina_Display.png/500px-MacBook_with_Retina_Display.png';

  function handleClick(e) {
    alert('곧 도착합니다!');
  }

  ReactDOM.render(
    <>
      <h1>{product + ' ' + model} 주문하기</h1>
      <img src={imageUrl} alt="제품 사진" />
      <button onClick={handleClick}>확인</button>
    </>,
    document.getElementById('root')
  );
  ```

### Step 2-5. React Element and Component

- **react element** : JSX 문법으로 작성한 하나의 요소(javascript object)
- ReactDOM.rende 함수로 해석해서 HTML 형태로 브라우저에 띄움

  ```javascript
  import ReactDOM from 'react-dom';

  const element = <h1>안녕 리액트!</h1>;
  console.log(element);
  ReactDOM.render(element, document.getElementById('root'));
  // log : {$$typeof: Symbol(react.element), type: "h1", key: null, ref: null, props: {…}, …}
  ```

- **react component** : react element를 자유롭게 다루기 위한 하나의 문법
- 간단한 방법으로 javascript function 사용하며, 첫 글자를 **대문자**로 작성해야 함
- 아래 코드에서 JSX 문법으로 작성된 하나의 요소를 리턴하는 Hello 함수가 하나의 컴포넌트

  ```javascript
  function Hello() {
  return <h1>안녕 리액트</h1>;
  }

  const element = (
    <>
      <Hello />
      <Hello />
      <Hello />
    </>
  );

  ReactDOM.render(element, document.getElementById('root'));
  ```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. 

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. 