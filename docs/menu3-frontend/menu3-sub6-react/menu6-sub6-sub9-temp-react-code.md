---
layout: default
title: React Temp Code
parent: React
grand_parent: Frontend
nav_order: 9
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

### codeit react basic project

## STEP 1. 주사위게임

### Step 1-1. ch1
- **`bash`**


### Step 1-2. 

<!------------------------------------ STEP ------------------------------------>

## STEP 2.가위바위보게임

### Step 2-1. 기본 HTML 코드를 JSX로 바꿔서 기본요소 배치하기

* **`public/index.html`**

  ```html
  <!DOCTYPE html>
  <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <title>가위바위보</title>
    </head>
    <body>
      <div id="root"></div>
    </body>
  </html>
  ```

* **`src/index.js`**

  ```javascript
  import ReactDOM from 'react-dom';

  ReactDOM.render(<div id='root'>
  <h1 id="title">가위바위보</h1>
  <button className="hand">가위</button>
  <button className="hand">바위</button>
  <button className="hand">보</button>
    </div>
  , document.getElementById('root'));
  ```

### Step 2-2. ch2
