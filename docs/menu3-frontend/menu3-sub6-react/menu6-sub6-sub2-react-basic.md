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

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. 

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. 