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

### Step 1-1. 기본 배치 및 컴포넌트

* **`public/index.html`**

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

* **`scr/index.js`**

  ```javascript
  import ReactDOM from 'react-dom';
  import App from './App';

  ReactDOM.render(<App />, document.getElementById('root'));
  ```

* **`src/App.js`**

  ```javascript
  import Dice from './Dice';

  function App() {
    return (
      <div>
        <Dice />
      </div>
    );
  }

  export default App;
  ```

* **`scr/Dics.js`**

  ```javascript
  import diceBlue01 from './assets/dice-blue-1.svg';

  function Dice() {
    return <img src={diceBlue01} alt="주사위" />;
  }

  export default Dice;
  ```

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

### Step 2-2. JSX에서 자바스크립트를 추가

* **`src/index.js`**

  ```javascript
  import ReactDOM from 'react-dom';

  const WINS = {
    rock: 'scissor',
    scissor: 'paper',
    paper: 'rock',
  };

  function getResult(left, right) {
    if (WINS[left] === right) return '승리';
    else if (left === WINS[right]) return '패배';
    return '무승부';
  }

  function handleClick() {
    console.log('가위바위보!');
  }

  const me = 'rock';
  const other = 'scissor';

  ReactDOM.render(
    <>
      <h1>가위바위보</h1>
      <h2>{getResult(me, other)}</h2>
      <button onClick={handleClick}>가위</button>
      <button onClick={handleClick}>바위</button>
      <button onClick={handleClick}>보</button>
    </>,
    document.getElementById('root')
  );
  ```

