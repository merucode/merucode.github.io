---
layout: default
title: React Summary
parent: React
grand_parent: Frontend
nav_order: 1
---

# React Summary
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

## STEP 1. React Basic

### Step 1-1. **`bash`**

```bash
$             # node.js 설치
$ node -v     # node.js 버전 확인 
$ npm -v      # npm 버전 확인

$ npm init react-app .    # 현재 폴더에서 react 프로젝트 시작
$ npm run start           # 개발서버 실행(localhost:3000)

$ npm run build           # react 프로젝트 빌드(create build folder)
$ npx server build        # npm 저장소에서 server 프로그램 다운 후 build 폴더에서 서버 실행(localhost:5000)

```

### Step 1-2. **`.js`**

```react
//---------- 기본 구성 ----------//
import ReactDOM from 'react-dom/client';
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<h1>안녕 리액트!</h1>);   // render 안에 JSX

            
//---------- Fragment ----------//
root.render(
  <>
  <h1>안녕 리액트!</h1>
  <h1>안녕 리액트!</h1>
  </>
); 


//---------- Component ----------// 
function Hello() {
return <h1>안녕 리액트</h1>;
}

const element = (
  <>
    <Hello />
    <Hello />
  </>
);


//---------- Props ----------// 
/* Dics.js */
function Dice({ color = 'blue', num = 1 }) {
  const src = DICE_IMAGES[color][num - 1];
  const alt = `${color} ${num}`;
  return <img src={src} alt={alt} />;
}
export default Dice;

/* App.js */
import Dice from './Dice';
...
<Dice color='blue' num={num} />


//---------- Chiledren ----------//
/* Button.js */
function Button({ children, onClick }) {
  return <button onClick={onClick}>{children}</button>;
}
export default Button;

/* App.js */
import Button from './Button';
...
<Button onClick={handleRollClick}>던지기</Button>


//---------- State ----------//
import { useState } from 'react';
const [num, setNum] = useState(1);
setNum(1);

//reference type state
const [gameHistory, setGameHistory] = useState([]);
setGameHistory([...gameHistory, 1]); 


//---------- className----------//
/* App.js */
import './App.css';
...
<Button className="App-button" onClick={handleRollClick}>던지기</Button>

/* Button.js */
import './Button.css';
function Button({ className = '', color = 'blue', children, onClick }) {
  const classNames = `Button ${color} ${className}`;
  return (
  <button className={classNames} onClick={onClick}>
    {children}
  </button>
  );
}
```



<br>



## STEP 9. utils.js

```react
/*  1~n 사이 랜덤한 정수 반환 */
function random(n) {
  return Math.ceil(Math.random() * n);
}

```