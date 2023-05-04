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

  ```react
  import ReactDOM from 'react-dom';
  import App from './App';

  ReactDOM.render(<App />, document.getElementById('root'));
  ```

* **`src/App.js`**

  ```react
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

  ```react
  import diceBlue01 from './assets/dice-blue-1.svg';

  function Dice() {
    return <img src={diceBlue01} alt="주사위" />;
  }

  export default Dice;
  ```

### Step 1-2. [Props/children 적용](https://www.codeit.kr/learn/4736)

* **`src/App.js`**

  ```react
  import Button from './Button';
  import Dice from './Dice';

  function App() {
    return (
      <div>
        <Button>던지기</Button>
        <Button>처음부터</Button>
        <Dice color="red" num={4} />
      </div>
    );
  }

  export default App;
  ```

* **`scr/Dics.js`**

  ```react
  import diceBlue01 from './assets/dice-blue-1.svg';
  import diceBlue02 from './assets/dice-blue-2.svg';
  import diceBlue03 from './assets/dice-blue-3.svg';
  import diceBlue04 from './assets/dice-blue-4.svg';
  import diceBlue05 from './assets/dice-blue-5.svg';
  import diceBlue06 from './assets/dice-blue-6.svg';
  import diceRed01 from './assets/dice-red-1.svg';
  import diceRed02 from './assets/dice-red-2.svg';
  import diceRed03 from './assets/dice-red-3.svg';
  import diceRed04 from './assets/dice-red-4.svg';
  import diceRed05 from './assets/dice-red-5.svg';
  import diceRed06 from './assets/dice-red-6.svg';

  const DICE_IMAGES = {
    blue: [diceBlue01, diceBlue02, diceBlue03, diceBlue04, diceBlue05, diceBlue06],
    red: [diceRed01, diceRed02, diceRed03, diceRed04, diceRed05, diceRed06],
  };

  function Dice({ color = 'blue', num = 1 }) {
    const src = DICE_IMAGES[color][num - 1];
    const alt = `${color} ${num}`;
    return <img src={src} alt={alt} />;
  }

  export default Dice;
  ```

* **`src/Button.js`**

  ```react
  function Button({ children }) {
    return <button>{children}</button>;
  }

  export default Button;
  ```

### Step 1-3. [state 적용](https://www.codeit.kr/learn/4740)

* **`src/App.js`**
  ```react
  import { useState } from 'react';
  import Button from './Button';
  import Dice from './Dice';

  function random(n) {
    return Math.ceil(Math.random() * n);
  }

  function App() {
    const [num, setNum] = useState(1);
    const [sum, setSum] = useState(0);
    const [gameHistory, setGameHistory] = useState([]);

    const handleRollClick = () => {
      const nextNum = random(6);
      setNum(nextNum);
      setSum(sum + nextNum);
      setGameHistory([...gameHistory, nextNum]);
    };

    const handleClearClick = () => {
      setNum(1);
      setSum(0);
      setGameHistory([]);
    };

    return (
      <div>
        <div>
          <Button onClick={handleRollClick}>던지기</Button>
          <Button onClick={handleClearClick}>처음부터</Button>
        </div>
        <div>
          <h2>나</h2>
          <Dice color='blue' num={num} />
          <h2>총점</h2>
          <p>{sum}</p>
          <h2>기록</h2>
          {gameHistory.join(', ')}
        </div>
      </div>
    );
  }

  export default App;
  ```


### Step 1-4. [컴포넌트 재사용하기](https://www.codeit.kr/learn/4744)

### Step 1-5. [코드 정리하기](https://www.codeit.kr/learn/4745)

### Step 1-6. [CSS 클래스네임 적용](https://www.codeit.kr/learn/4749)

### Step 1-7. [마무리 디자인](https://www.codeit.kr/learn/4658)













<!------------------------------------ STEP ------------------------------------>

## STEP 2.가위바위보게임

### Step 2-1. 기본 HTML 코드를 JSX로 바꿔서 기본요소 배치하기

### Step 2-2. JSX에서 자바스크립트를 추가

### Step 2-3. (가위바위보 핸드아이콘 적용_props)[https://www.codeit.kr/learn/4734]

### Step 2-4. (가위바위보 버튼 적용_props)[https://www.codeit.kr/learn/4735]

### Step 2-5. (초기화 버튼 생성_children)[https://www.codeit.kr/learn/4737]

### Step 2-6. (가위바위보 승부기록_state)[https://www.codeit.kr/learn/4741]

### Step 2-7. (가위바위보 배점_state)[https://www.codeit.kr/learn/4742]

### Step 2-8. (가위바위보 클래스네임 적용_style)[https://www.codeit.kr/learn/4750]

### Step 2-9. [마무리 디자인](https://www.codeit.kr/learn/4658)