---
layout: default
title: React Note
parent: React
grand_parent: Frontend
nav_order: 9
---

# React Note
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

## STEP 1. useState


```react
// 초기값 지정하기
import { useState } from 'react';
const [state, setState] = useState(initialState);



// 콜백으로 초기값 지정하기
const [state, setState] = useState(() => {
  // 초기값을 계산
  return initialState;
});

// 예문
function ReviewForm() {
  const [values, setValues] = useState(() => {
    const savedValues = getSavedValues(); // 처음 렌더링할 때만 실행됨
    return savedValues
});



// Setter 함수 사용하기
const [state, setState] = useState(0);

const handleAddClick = () => {
  setState(state + 1);
}

// 참조형 state
const [state, setState] = useState({ count: 0 });

const handleAddClick = () => {
  setState({ ...state, count: state.count + 1 }); // 새로운 객체 생성
}



// 콜백으로 State 변경(비동기시 사용)
setState((prevState) => {
  // 다음 State 값을 계산
  return nextState;
});

// 예문
const [count, setCount] = useState(0);

const handleAddClick = async () => {
  await addCount();
  setCount((prevCount) => prevCount + 1);
}
```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. useRef

```react
// Ref 객체 생성
import { useRef } from 'react';
const ref = useRef();


// Ref 객체에서 DOM 노드 참조하기
const node = ref.current;
if (node) {
  // node 를 사용하는 코드
}
/* Ref 객체의 current 라는 프로퍼티를 사용하면 DOM 노드를 참조할 수 있었습니다.
current 값은 없을 수도 있으니까 반드시 값이 존재하는지 검사하고 사용해야 하는 점도 잊지 마세요!*/


// 예시: 이미지 크기 구하기
import { useRef } from 'react';

function Image({ src }) {
  const imgRef = useRef();

  const handleSizeClick = () => {
    const imgNode = imgRef.current;
    if (!imgNode) return;

    const { width, height } = imgNode;
    console.log(`${width} x ${height}`);
  };

  return (
    <div>
      <img src={src} ref={imgRef} alt="크기를 구할 이미지" />
      <button onClick={handleSizeClick}>크기 구하기</button>
    </div>
  );
}
```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. 사이드 이펙트(Side Effect)와 useEffect

### Step 3-1. useEffect 기본

* Side Effect : 외부에 부수적인 작용

```react
// 예시
let count = 0;
function add(a, b) {
  const result = a + b;
  count += 1; // 함수 외부의 값을 변경(사이드 이펙트)
  return result;
}
const val1 = add(1, 2);
const val2 = add(-4, 5)


// 사이드 이펙트와 useEffect
// useEffect : 리액트 외부에 있는 데이터나 상태를 변경할 때 사용
// DOM 노드 직접 변경, 브라우저 데이터 저장,네트워크 리퀘스트 등

// 페이지 정보 변경
useEffect(() => {
  document.title = title; // 페이지 데이터를 변경
}, [title]);

// 네트워크 요청
useEffect(() => {
  fetch('https://example.com/data') // 외부로 네트워크 리퀘스트
    .then((response) => response.json())
    .then((body) => setData(body));
}, [])

// 데이터 저장
useEffect(() => {
  localStorage.setItem('theme', theme); // 로컬 스토리지에 테마 정보를 저장
}, [theme]);

// 타이머
useEffect(() => {
  const timerId = setInterval(() => {
    setSecond((prevSecond) => prevSecond + 1);
  }, 1000); // 1초마다 콜백 함수를 실행하는 타이머 시작
  
  return () => {
    clearInterval(timerId);
  }
}, []);
```

### Step 3-2. useEffect 장점(동기화)
  * '동기화'에 쓰면 유용한 경우
  * 컴포넌트 안에 데이터와 리액트 바깥에 있는 데이터를 일치

```react
import { useEffect, useState } from 'react';

const INITIAL_TITLE = 'Untitled';

function App() {
  const [title, setTitle] = useState(INITIAL_TITLE);

  const handleChange = (e) => {
    const nextTitle = e.target.value;
    setTitle(nextTitle);
  };

  const handleClearClick = () => {
    setTitle(INITIAL_TITLE);
  };

  useEffect(() => {
    document.title = title;
  }, [title]);

  return (
    <div>
      <input value={title} onChange={handleChange} />
      <button onClick={handleClearClick}>초기화</button>
    </div>
  );
}

export default App;
```

* `document.title` 값과 `state title` 값을 쉽게 일치 가능

### Step 3-3. 정리 함수(Cleanup Function)

* 정리 함수가 실행되는 시점 : 쉽게 말해서 콜백을 한 번 실행했으면, 정리 함수도 반드시 한 번 실행
* 정확히는 새로운 콜백 함수가 호출되기 전에 실행되거나 (앞에서 실행한 콜백의 사이드 이펙트를 정리), 컴포넌트가 화면에서 사라지기 전에 실행됩니다 (맨 마지막으로 실행한 콜백의 사이드 이펙트를 정리)

```react
useEffect(() => {
  // 사이드 이펙트
  return () => {
    // 사이드 이펙트에 대한 정리
  }
}, [dep1, dep2, dep3, ...]);

/* 예를 들면 이미지 파일 미리보기를 구현할 때 Object URL을 만들어서 브라우저의 메모리를 할당(createObjectURL) 했는데요. 정리 함수에서는 이때 할당한 메모리를 다시 해제(revokeObjectURL)해줬었죠.
*/
```

```react
// 예시 : 타이머
import { useEffect, useState } from 'react';

function Timer() {
  const [second, setSecond] = useState(0);

  useEffect(() => {
    const timerId = setInterval(() => {
      console.log('타이머 실행중 ... ');
      setSecond((prevSecond) => prevSecond + 1);
    }, 1000);
    console.log('타이머 시작 🏁');

    return () => {
      clearInterval(timerId);
      console.log('타이머 멈춤 ✋');
    };
  }, []);

  return <div>{second}</div>;
}

function App() {
  const [show, setShow] = useState(false);

  const handleShowClick = () => setShow(true);
  const handleHideClick = () => setShow(false);

  return (
    <div>
      {show && <Timer />}
      <button onClick={handleShowClick}>보이기</button>
      <button onClick={handleHideClick}>감추기</button>
    </div>
  );
}

export default App;
```