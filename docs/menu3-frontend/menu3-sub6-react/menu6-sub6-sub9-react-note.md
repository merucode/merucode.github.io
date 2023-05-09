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

## Step 1. usestate


```react
// 초기값 지정하기
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