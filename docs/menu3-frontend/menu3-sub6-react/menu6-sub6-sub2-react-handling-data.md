---
layout: default
title: React Handling Data
parent: React
grand_parent: Frontend
nav_order: 2
---

# React Handling Data
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

## STEP 1. Randering Array

### Step 1-1. mock 데이터 추가하기

- mock 데이터 : 네트워크에서 받아올 데이터를 흉내낸 데이터

  ```react
  /* App.js */
    import ReviewList from './ReviewList';
    import items from '../mock.json';

    function App() {
      return (
        <div>
          <ReviewList items={items} />
        </div>
      );
    }

    export default App;

  /* ReviewList.js */
    function ReviewList({ items }) {
    console.log(items);
    return <ul></ul>;
    }

    export default ReviewList;
  ```

### Step 1-2. map으로 배열 렌더링하기

* [pokemon.json](https://bakey-api.codeit.kr/api/files/resource?root=static&seqId=5035&directory=pokemons.json&name=pokemons.json)
* 배열 메소드 map에서 콜백 함수의 리턴 값으로 리액트 엘리먼트를 리턴

  ```react
  import items from './pokemons';

  function Pokemon({ item }) {
    return (
      <div>
        No.{item.id} {item.name}
      </div>
    );
  }

  function App() {
    return (
      <ul>
        {items.map((item) => (
          <li key={item.id}>
            <Pokemon item={item} />
          </li>
        ))}
      </ul>
    );
  }
  
  export default App;
  ```

### Step 1-3. sort로 배열하기

* 배열 메소드의 sort 메소드를 사용. 아래 코드는 id 순서대로 / 반대로 정렬하는 예시

  ```react
  import { useState } from 'react';
  import items from './pokemons';

  function Pokemon({ item }) {
    return (
      <div>
        No.{item.id} {item.name}
      </div>
    );
  }

  function App() {
    const [direction, setDirection] = useState(1);

    const handleAscClick = () => setDirection(1);

    const handleDescClick = () => setDirection(-1);

    const sortedItems = items.sort((a, b) => direction * (a.id - b.id));

    return (
      <div>
        <div>
          <button onClick={handleAscClick}>도감번호 순서대로</button>
          <button onClick={handleDescClick}>도감번호 반대로</button>
        </div>
        <ul>
          {sortedItems.map((item) => (
            <li key={item.id}>
              <Pokemon item={item} />
            </li>
          ))}
        </ul>
      </div>
    );
  }

  export default App;
  ```

### Step 1-4. filter로 아이템 삭제하기

* 배열 메소드 중 filter 와 배열형 스테이트를 활용하여 삭제 기능 구현

  ```react
  import { useState } from 'react';
  import mockItems from './pokemons';

  function Pokemon({ item, onDelete }) {
    const handleDeleteClick = () => onDelete(item.id);

    return (
      <div>
        No.{item.id} {item.name}
        <button onClick={handleDeleteClick}>삭제</button>
      </div>
    );
  }

  function App() {
    const [items, setItems] = useState(mockItems);

    const handleDelete = (id) => {
      const nextItems = items.filter((item) => item.id !== id);
      setItems(nextItems);
    };

    return (
      <ul>
        {items.map((item) => (
          <li key={item.id}>
            <Pokemon item={item} onDelete={handleDelete} />
          </li>
        ))}
      </ul>
    );
  }

  export default App;
  ```

### Step 1-5. 배열 렌더링 시 key 사용

* 배열 렌더링 시 key를 지정해줘야 Console Warning 경고 발생 안함
* 요소들의 순서가 바뀔 때 key가 없다면 엉뚱한 위치로 렌더링 될 수 있음
* **key는 요소들의 고유한 값**으로 지정해줘야 함
* 렌더링 가장 바깥쪽에 있는 (최상위) 태그에다가 key Prop을 지정하며, 반드시 id 일 필요는 없고 포켓몬 이름처럼(참고로 포켓몬 이름은 고유합니다) 각 데이터를 구분할 수 있는 고유한 값이면 무엇이든 key 로 활용 가능


  ```react
  ...
  // CH 1-2. mapping rendering
  // CH 2-2. Add filter(onDelete)
  // CH 2-3. Add key(item.id)
  function ReviewList({ items, onDelete }) {
    return (
      <ul>
        {items.map((item) => {
          return (
            <li key={item.id}>
              <ReviewListItem item={item} onDelete={onDelete} />
            </li>
          );
        })}
      </ul>
    );
  }
  ...
  ```


<br> 

<!------------------------------------ STEP ------------------------------------>
## STEP 2. Load Data

### Step 2-1. fetch

* 실습 서버 주소 : https://learn.codeit.kr/2001/film-reviews/

* 예문

  ```react
  /* api.js */
  export async function getReviews() {
    const response = await fetch('https://learn.codeit.kr/api/film-reviews');
    const body = await response.json();
    return body;
  }

  /* App.js */
  import { getReviews } from '../api';
  ...
  function App() {
    const [items, setItems] = useState([]);
    ...
    const handleLoadClick = async () => {
      const { reviews } = await getReviews();
      setItems(reviews);
    };
  ...
  }
  ```

### Step 2-2. useEffect

* **처음 한 번만 실행하기** : 컴포넌트가 처음 렌더링 되고 나면 리액트가 콜백 함수를 기억해뒀다가 실행. 그 이후로는 콜백 함수를 실행하지 않습니다.

  ```react
  useEffect(() => {
    // 실행할 코드
  }, []);
  ```

* **값이 바뀔 때마다 실행하기** : 디펜던시 리스트에 있는 값들을 확인해서 하나라도 바뀌면  콜백 함수를 기억해뒀다가 실행

  ```react
  useEffect(() => {
    // 실행할 코드
  }, [dep1, dep2, dep3, ...]);
  ```

* 예문

  ```react
  import { useEffect, useState } from 'react';

  function App() {
    const [first, setFirst] = useState(1);
    const [second, setSecond] = useState(1);

    const handleFirstClick = () => setFirst(first + 1);
    const handleSecondClick = () => setSecond(second + 1);

    useEffect(() => {
      console.log('렌더링 이후', first, second);
    }, [first]);
    // 디펜던시 리스트에 [] , [first], [first, second] 바꾸어가며 확인

    console.log('렌더링', first, second);

    return (
      <div>
        <h1>
          {first}, {second}
        </h1>
        <button onClick={handleFirstClick}>First</button>
        <button onClick={handleSecondClick}>Second</button>
      </div>
    );
  }

  export default App;
  ```

### Step 2-3. Pagination

* Pagination : 책의 페이지처럼 데이터를 나눠서 제공하는 것
  * 오프셋 기반, 커서 기반

* 오프셋(Offset) 기반 : 받아온 데이터 갯수를 기준으로 데이터를 나눔 → 받아오는 중간에 데이터 추가/삭제 시 중복, 결실 발생 → 커서 기반 사용

* 커서(Cursor) 기반 : 특정 데이터(책갈피) 기준

|Items|url|
|---|---|
|Offset|[1:42](https://www.codeit.kr/learn/5044)|
|Cursur|[3:45](https://www.codeit.kr/learn/5044)|

### Step 2-3. 오프셋 기반

* 예문

  ```react
  /* App.js */
  import { getReviews } from '../api';
  ...
  const LIMIT = 6; // pagination limit

  function App() {
    const [items, setItems] = useState([]);
    const [offset, setOffset] = useState(0);          // pagination offset
    const [hasNext, setHasNext] = useState(false);    // pagination 마지막 페이지 확인

    const handleDelete = (id) => {...};

    const handleLoad = async (options) => {
      const { reviews, paging } = await getReviews(options);  // getReviews response.json()의 구성을 보면 reviews, paging 존재
      if (options.offset === 0) {
        setItems(reviews);
      } else {
        setItems([...items, ...reviews]);   // 기존 data에 새로 불러온 데이터 추가 
      }
      setOffset(options.offset + reviews.length);
      setHasNext(paging.hasNext);           // 마지막 페이지시 더보기 버튼 안보이는 기능
    };

    const handleLoadMore = () => {
      handleLoad({ order, offset, limit:LIMIT });
    };

    useEffect(() => {
      handleLoad({ order, offset:0, limit:LIMIT });
    }, [order]);

    return (
      <div>
        <div>
          <button onClick={handleNewestClick}>최신순</button>
          <button onClick={handleBestClick}>베스트순</button>
        </div>
        <ReviewList items={sortedItems} onDelete={handleDelete} />
        {hasNext && <button onClick={handleLoadMore}>더 보기</button>}
      </div>
    );
  }

  export default App;

  /* api.js */
  export async function getReviews({ order = 'createdAt', offset = 0, limit = 6,}) {
    const query = `order=${order}&offset=${offset}&limit=${limit}`;
    const response = await fetch(
      `https://learn.codeit.kr/api/film-reviews?${query}`
    );
    const body = await response.json();
    return body;
  }
  ```

### Step 2-4. 커서 기반 

### Step 2-5. 조건부 렌더링

* 예문

  ```react
  import { useState } from 'react';

  function App() {
    const [show, setShow] = useState(false);

    const handleClick = () => setShow(!show);

    return (
      <div>
        <button onClick={handleClick}>토글</button>
        {show && <p>보인다 👀</p>}
        {show || <p>보인다 👀</p>} 
        {show ? <p>✅</p> : <p>❎</p>}
      </div>
    );
  }
  // && show 값이 true면 렌더링 O, false면 렌더링 X
  // || show 값이 true면 렌더링 X, false면 렌더링 O 
  // 삼항연산자 show 값이 true면 V, false면 X 렌더링

  export default App;
  ```

* 렌더링되지 않는 값들

  ```react
  const nullValue = null;
  const undefinedValue = undefined;
  const trueValue = true;
  const falseValue = false;
  const emptyString = '';
  const emptyArray = [];

  const zero = 0; // false과 동시에 0 렌더링 
  const one = 1;  // true 과 동시에 1 렌더링
  ```

* 조건부 렌더링 주의점

  ```react
    {num && <p>num이 0 보다 크다!</p>}        // num이 0일 경우 0이 같이 렌더링 됨
    {(num > 0) && <p>num이 0 보다 크다!</p>}  // 다음과 같이 명확한 조건문 사용
  ```

### Step 2-6. 비동기 state 변경시 주의점

만약 이전 State 값을 참조하면서 State를 변경하는 경우,
비동기 함수에서 State를 변경하게 되면 최신 값이 아닌 State 값을 참조하는 문제가 있었습니다.(변경 중 데이터 삭제 등 작업 시 미반영됨)
이럴 때는 콜백을 사용해서 처리할 수 있었는데요. 파라미터로 올바른 State 값을 가져와서 사용할 수 있습니다.
이전 State 값으로 새로운 State를 만드는 경우엔 항상 콜백 형태를 사용하는 습관 사용

  ```react
  const [count, setCount] = useState(0);

  const handleAddClick = async () => {
    await addCount();
    setCount((prevCount) => prevCount + 1); // 비동기 state 변경 시 콜백 형태 사용
  }
  ```

### Step 2-7. 네트워크 로딩 처리

* 예문

  ```react
  function App() {
  ...
  const [isLoading, setIsLoading] = useState(false);
  ...
  const handleLoad = async (options) => {
    let result;
    try {                                  // 로딩 처리
      setIsLoading(ture);
      result = await getReviews(options);  
    } catch (error) {
      console.error(error);
      return;
    } finally {
      setIsLoading(false);
    }
  ...
  return (...
    {hasNext && <button disabled={isLoading} onClick={handleLoadMore}>더 보기</button>}
  ...);
  }
  ```

### Step 2-8. 네트워크 에러 처리

* 예문

  ```react
  /* App.js */
  function App() {
  ...
  const [loadingError, setLoadingError] = useState(null); // CH 7-2. 에러 처리
  ...
  const handleLoad = async (options) => {
    let result;
    try {                                 // CH 7-1. 로딩 처리
      setIsLoading(true);
      setLoadingError(null);
      result = await getReviews(options);  
    } catch (error) {
      setLoadingError(error);             // CH 7-2. 에러 처리
      return;
    } finally {
      setIsLoading(false);
    }
  ...
  return (...
      {loadingError?.message && <span>{loadingError.message}</span>}
  ...);
  }

  /* api.js */
  export async function getReviews({
    order = 'createdAt',
    offset = 0,
    limit = 6,
  }) {
    const query = `order=${order}&offset=${offset}&limit=${limit}`;
    const response = await fetch(
      `https://learn.codeit.kr/api/film-reviews?${query}`
    );
    if (!response.ok) {     // CH 7-2. 에러 처리
      throw new Error('리뷰를 불러오는데 실패했습니다');
    }
    const body = await response.json();
    return body;
  }
  ```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. Input Form

### Step 3-1. HTML과 다른 점

* <input>의 `onChange`
  * 리액트에선 순수 HTML과 다르게 onChange Prop을 사용하면 입력 값이 바뀔 때마다 핸들러 함수를 실행(HTML `oninput` 이벤트와 같다고 생각)

* `htmlFor`
  * <label /> 태그에서 사용하는 속성인 `for` 는 자바스크립트 반복문 키워드인 `for` 와 겹치기 때문에 리액트에서는 `htmlFor`를 사용

### Step 3-2. 폼을 다루는 기본적인 방법

* 스테이트를 만들고 `target.value` 값을 사용해서 값을 변경

  ```react
  function TripSearchForm() {
    const [location, setLocation] = useState('Seoul');
    const [checkIn, setCheckIn] = useState('2022-01-01');
    const [checkOut, setCheckOut] = useState('2022-01-02');

    const handleLocationChange = (e) => setLocation(e.target.value);

    const handleCheckInChange = (e) => setCheckIn(e.target.value);

    const handleCheckOutChange = (e) => setCheckOut(e.target.value);
      
    return (
      <form>
        <h1>검색 시작하기</h1>
        <label htmlFor="location">위치</label>
        <input id="location" name="location" value={location} placeholder="어디로 여행가세요?" onChange={handleLocationChange} />
        <label htmlFor="checkIn">체크인</label>
        <input id="checkIn" type="date" name="checkIn" value={checkIn} onChange={handleCheckInChange} />
        <label htmlFor="checkOut">체크아웃</label>
        <input id="checkOut" type="date" name="checkOut" value={checkOut} onChange={handleCheckOutChange} />
        <button type="submit">검색</button>
      </form>
    )
  }
  ```

### Step 3-3. 폼 값을 객체 하나로 처리하기

* 이벤트 객체의 `target.name` 과 `target.value` 값을 사용해서 값을 변경

  ```react
  function TripSearchForm() {
    const [values, setValues] = useState({
      location: 'Seoul',
      checkIn: '2022-01-01',
      checkOut: '2022-01-02',
    })

    const handleChange = (e) => {
      const { name, value } = e.target;
      setValues((prevValues) => ({
        ...prevValues,
        [name]: value,
      }));
    }
      
    return (
      <form>
        <h1>검색 시작하기</h1>
        <label htmlFor="location">위치</label>
        <input id="location" name="location" value={values.location} placeholder="어디로 여행가세요?" onChange={handleChange} />
        <label htmlFor="checkIn">체크인</label>
        <input id="checkIn" type="date" name="checkIn" value={values.checkIn} onChange={handleChange} />
        <label htmlFor="checkOut">체크아웃</label>
        <input id="checkOut" type="date" name="checkOut" value={values.checkOut} onChange={handleChange} />
        <button type="submit">검색</button>
      </form>
    )
  }
  ```

### Step 3-4. 기본 submit 동작 막기

* HTML 폼의 기본 동작은 submit 타입의 버튼을 눌렀을 때 페이지를 이동하는 건데요. 이벤트 객체의 preventDefault 를 사용하면 이 동작을 막을 수 있었습니다.

  ```react
  const handleSubmit = (e) => {
    e.preventDefault();
    // ...
  }
  ```

### Step 3-5. 제어 컴포넌트(권장)

* 인풋 태그의 `value` 속성을 지정하고 사용하는 컴포넌트
* 리액트에서 지정한 값과 실제 인풋 value 의 값이 항상 같음
* State냐 Prop이냐는 중요하지 않고, 리액트로 value 를 지정한다는 것이 핵심

  ```react
  function TripSearchForm() {
    const [values, setValues] = useState({
      location: 'Seoul',
      checkIn: '2022-01-01',
      checkOut: '2022-01-02',
    })

    const handleChange = (e) => {
      const { name, value } = e.target;
      setValues((prevValues) => ({
        ...prevValues,
        [name]: value,
      }));
    }
      
    return (
      <form>
        <h1>검색 시작하기</h1>
        <label htmlFor="location">위치</label>
        <input id="location" name="location" value={values.location} placeholder="어디로 여행가세요?" onChange={handleChange} />
        <label htmlFor="checkIn">체크인</label>
        <input id="checkIn" type="date" name="checkIn" value={values.checkIn} onChange={handleChange} />
        <label htmlFor="checkOut">체크아웃</label>
        <input id="checkOut" type="date" name="checkOut" value={values.checkOut} onChange={handleChange} />
        <button type="submit">검색</button>
      </form>
    )
  }
  ```

### Step 3-6. 비제어 컴포넌트

* 인풋 태그의 `value` 속성을 리액트에서 지정하지 않고 사용하는 컴포넌트
* 파일 선택 인풋 등에 사용

  ```react
  function TripSearchForm({ onSubmit }) {
    return (
      <form onSubmit={onSubmit} >
        <h1>검색 시작하기</h1>
        <label htmlFor="location">위치</label>
        <input id="location" name="location" placeholder="어디로 여행가세요?" />
        <label htmlFor="checkIn">체크인</label>
        <input id="checkIn" type="date" name="checkIn" />
        <label htmlFor="checkOut">체크아웃</label>
        <input id="checkOut" type="date" name="checkOut" />
        <button type="submit">검색</button>
      </form>
    )
  }

  // 폼 태그는 참조 가능
  const handleSubmit = (e) => {
    e.preventDefault();
    const form = e.target;
    const location = form['location'].value;
    const checkIn = form['checkIn'].value;
    const checkOut = form['checkOut'].value;
    // ....
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    const form = e.target;
    const formValue = new FormValue(form);
    // ...
  }
  ```

### Step 3-7. File Input

* 비제어 컴포넌트로 만들어야

  ```react
  /* FileInput.js */
  function FileInput({ name, value, onChange}) {
      const handleChange = (e) => {
          const nextValue = e.target.files[0];
          onChange(name, nextValue);
      };

      return <input type="file" onChange={handleChange} />;
      // props에 value={value} 넣으면 비제어 input이라는 이유로 경보 발생
      // file input은 반드시 비제어 컴포넌트로 만들어야
  }

  export default FileInput;


  /* ReviewForm.js */
  import FileInput from './FileInput';
  ...
  function ReviewForm() {
    const [values, setValues] = useState({    
      title: '',
      rating: 0,
      content: '',
      imgFile: null,
    });

    const handleChange = (name, value) => {
      setValues((prevValues) => ({
        ...prevValues,
        [name]: value,
      }));
    };

    const handleInputChange = (e) => {            
      const { name, value } = e.target;
      handleChange(name, value);
    };

    return (
      <form className="ReviewForm" onSubmit={handleSubmit}>
        <FileInput name="imgFile" value={values.imgFile} onChange={handleChange} />
        ...
      </form>
    );
  }
  ```

### Step 3-8. Clear File Input

  ```react
  /* FileInput.js */
  // CH 11. Add file input clear(useRef)
  import { useRef } from "react";

  // CH 10. Add file input
  // CH 11. Add file input clear(useRef)
  function FileInput({ name, value, onChange}) {
    const inputRef = useRef();

    const handleChange = (e) => {
        const nextValue = e.target.files[0];
        onChange(name, nextValue);
    };

    // CH 11. Clear file input
    const handleClearClick = () => {
        const inputNode = inputRef.current;
        if (!inputNode) return;

        inputNode.value = '';
        onChange(name, null);
    }

    // CH 11. Clear file input
    return (
    <div>
        <input type="file" onChange={handleChange} ref={inputRef} />
        {value && <button onClick={handleClearClick}>X</button>}
    </div>
  );
  }
  ```

### Step 3-9. Preview Input File

  ```react
  const [preview, setPreview] = useState();
  
  useEffect(() => {
      if (!value) return;     // 값 없는 경우 처리

      const nextPreview = URL.createObjectURL(value);     // 미리보기를 위한 이미지 URL 생성
      setPreview(nextPreview);

      return () => {          // CH 12. 사이드 이펙트 메모리 할당 해제(정리)
          setPreview();       // setPrereview 빈값으로
          URL.revokeObjectURL(nextPreview);   // URL 설정 해제
      }
  }, [value]);

  return (...
      <img src={preview} alt="이미지 미리보기" />
  ...)
  ```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 4. Send Data

### Step 4-0. form data 보내기 예제

```react
const formData = new FormData();
formData.append('title', '라라랜드');
formData.append('rating', 5);
formData.append('content', '재미있다!');
fetch('https://learn.codeit.kr/api/film-reviews', {
  method: 'POST',
  body: formData,
});
```

### Step 4-1. Send Data 관련 submit 및 api 연동

```react
/* api.js */
...
export async function createReview(formData) {
  const response = await fetch(
    `${BASE_URL}/film-reviews`, {
      method: 'POST',
      body: formData,
    }
  );
  if (!response.ok) {     // 에러 처리
    throw new Error('리뷰를 생성하는데 실패했습니다');
  }
  const body = await response.json();
  return body;
}

/* ReviewForm.js */
import { createReview } from '../api';
...
const handleSubmit = async (e) => {      
  e.preventDefault();             
  const formData = new FormData();
  formData.append('title', values.title);
  formData.append('rating', values.rating);
  formData.append('content', values.content);
  formData.append('imgFile', values.imgFile);
  try {
    setSubmittingError(null);
    setIsSubmitting(true);
    await createReview(formData);
  } catch (error) {
    setSubmittingError(error);
    return;
  } finally {
    setIsSubmitting(false);
  }
  setValues(INITIAL_VALUES);
};
...
return (
  ...
  <button type="submit" disabled={isSubmitting}>확인</button>
  {submittingError?.message && <div>{submittingError.message}</div>}
  ...
)
```


### Step 4-2. Submit Data 새로고침 없이 반영하기

```react
/* App.js */
...
function App() {
  ...
  const handleSubmitSuccess = (review) => {
    setItems((prevItems) => [review, ...prevItems]);
  };
  ...
  return (
    ...
    <ReviewForm onSubmitSuccess={handleSubmitSuccess} />
    ...
  );
}

/* ReviewForm.js */
...
function ReviewForm({ onSubmitSuccess}) {
  ...
  const handleSubmit = async (e) => {      
    e.preventDefault(); 
                
    const formData = new FormData();
    formData.append('title', values.title);
    formData.append('rating', values.rating);
    formData.append('content', values.content);
    formData.append('imgFile', values.imgFile);
    
    let result;
    try {
      setSubmittingError(null);
      setIsSubmitting(true);
      result = await createReview(formData);
    } catch (error) {
      setSubmittingError(error);
      return;
    } finally {
      setIsSubmitting(false);
    }
    const { review } = result;
    onSubmitSuccess(review);
    setValues(INITIAL_VALUES);
  };
  ...
  return (
  ...
    <form className="ReviewForm" onSubmit={handleSubmit}>
    ...
    <button type="submit" disabled={isSubmitting}>확인</button>
    ...
    </form>
  ...
  )
}
```

### Step 4-3. 글 수정하기-1(수정화면 띄우기)

```react
/* ReviewLsit.js */ // 수정 화면 띄우기
import ReviewForm from './ReviewForm';
...
function ReviewList({ items, onDelete }) {
  const [editingId, setEditingId] = useState(null);
  const handleCancel = () => setEditingId(null);
  ...
  return (
    <ul>
      {items.map((item) => {
        if (item.id === editingId) {
          const { imgUrl, title, rating, content } = item;
          const initialValues = { title, rating, content };

          return (
            <li key={item.id}>
              <ReviewForm 
                initialValues={initialValues} 
                initialPreview={imgUrl}
                onCancel={handleCancel} 
              />
            </li>
          );
        }
        return (
          <li key={item.id}>
            <ReviewListItem item={item}
            onDelete={onDelete} 
            onEdit={setEditingId} 
            />
          </li>
        );
      })}
    </ul>
  );
}

function ReviewListItem({ item, onDelete, onEdit }) {
  ...
  const handleEditClick = () => {
    onEdit(item.id);
  };
  return (
    ...
    <button onClick={handleEditClick}>수정</button>
    ...
);
}

/* ReviewForm.js */ //수정 시 기본 값 보이게 하기 
...
function ReviewForm({ initialValues=INITIAL_VALUES, initialPreview, onSubmitSuccess, onCancel }) {
...
return (
  <FileInput name="imgFile" value={values.imgFile} initialPreview={initialPreview} onChange={handleChange} />
  ...
  {onCancel && <button onClick={onCancel}>취소</button>}
  ...
);
}

/* FileInput.js */  // 수정 시 이미지 미리보기
function FileInput({ name, value, initialPreview, onChange}) {
  const [preview, setPreview] = useState(initialPreview);
  ...
  useEffect(() => {
      if (!value) return;     // 값 없는 경우 처리

      const nextPreview = URL.createObjectURL(value);     // 미리보기를 위한 이미지 URL 생성
      setPreview(nextPreview);

      return () => {          // CH 12. 사이드 이펙트 메모리 할당 해제(정리)
          setPreview(initialPreview);         // 수정 이미지 미리보기
          URL.revokeObjectURL(nextPreview);   // URL 설정 해제
      }
  }, [value, initialPreview]);    // 수정 이미지 미리보기
  ...
}

```


### Step 4-4. 글 수정하기-2(update api 추가)

```react
/* ReviewForm.js */
function ReviewForm({ 
  initialValues=INITIAL_VALUES, 
  initialPreview, 
  onCancel, 
  onSubmit,
  onSubmitSuccess, 
}) {
  ...
  const handleSubmit = async (e) => {      
      ...
      
      try {
        ...
        result = await onSubmit(formData);
      } catch (error) {
        ...
      }
      ...
      const { review } = result;
      onSubmitSuccess(review);
      setValues(INITIAL_VALUES);
    };
}

/* App.js */
import { createReview, getReviews, updateReview } from '../api';
...
function App() {
  ...
  const handleCreateSuccess = (review) => {
    setItems((prevItems) => [review, ...prevItems]);
  };

  const handleUpdateSuccess = (review) => {
    setItems((prevItems) => {
      const splitIdx = prevItems.findIndex((item) => item.id === review.id);
      return [
        ...prevItems.slice(0, splitIdx),
        review,
        ...prevItems.slice(splitIdx + 1),
      ]
    });
  };
  ...
  return (...
    <ReviewForm 
      onSubmit={createReview} 
      onSubmitSuccess={handleCreateSuccess} 
    />
    <ReviewList 
      items={sortedItems} 
      onDelete={handleDelete} 
      onUpdate={updateReview} 
      onUpdateSuccess={handleUpdateSuccess} 
    />
  ...
);
}

/* ReviewList.js */
...
function ReviewList({ items, onDelete, onUpdate, onUpdateSuccess }) {
  ...
  return (
    ...
    {items.map((item) => {
      if (item.id === editingId) {
        const { id, imgUrl, title, rating, content } = item;
        ...
        const handleSubmit = (formData) => onUpdate(id, formData);
        const handleSubmitSuccess = (review) => {
            onUpdateSuccess(review);
            setEditingId(null);
        };
        return (
            <li key={item.id}>
              <ReviewForm 
                initialValues={initialValues} 
                initialPreview={imgUrl}
                onCancel={handleCancel} 
                onSubmit={handleSubmit}
                onSubmitSuccess={handleSubmitSuccess}
              />
            </li>
          );
        }
        ...
    )}
  );
}



/* api.js */
...
export async function updateReview(id, formData) {
  const response = await fetch(`${BASE_URL}/film-reviews/${id}`, {
      method: 'PUT',
      body: formData,
    }
  );
  if (!response.ok) {     // 에러 처리
    throw new Error('리뷰를 수정하는데 실패했습니다');
  }
  const body = await response.json();
  return body;
}
```

### Step 4-5. 글 삭제하기(delete api 추가)

* 콘솔 예제

```javascript
fetch('https://learn.codeit.kr/api/film-reviews/43', {method: 'DELETE'});
```

* 코드

```react
/* api.js */
...
export async function deleteReview(id, formData) {
  const response = await fetch(`${BASE_URL}/film-reviews/${id}`, {
      method: 'DELETE',
    }
  );
  if (!response.ok) {     // 에러 처리
    throw new Error('리뷰를 삭제하는데 실패했습니다');
  }
  const body = await response.json();
  return body;
}

/* App.js */
import { createReview, getReviews, updateReview, deleteReview } from '../api';
...
function App() {
  ...
  const handleDelete = async (id) => {
    const result = await deleteReview(id);
    if (!result) return;  // 삭제가 성공한 경우에만 반영
    
    setItems((prevItems) => prevItems.filter((item) => item.id !== id));
  };
  ...
}
```

### Step 4-6. Custom Hook

```react
/* hooks/useAsync.js */
import { useState } from "react";

function useAsync(asyncFunction) {
    const [pending, setPending] = useState(false);
    const [error, setError] = useState(null);

    const wrappedFunction = async (...args) => {
        try {
            setError(null);
            setPending(true);
            return await asyncFunction(...args);
        } catch (error) {
            setError(error);
            return;
        } finally {
            setPending(false);
        }
    };

    return [pending, error, wrappedFunction];
}

export default useAsync;

/* App.js */
import { createReview, getReviews, updateReview, deleteReview } from '../api';
import useAsync from '../hooks/useAsync';
...
function App() {
  const [isLoading, loadingError, getReviewsAsync] = useAsync(getReviews);
  ...
  const handleLoad = async (options) => {
    const result = await getReviewsAsync(options);
    if (!result) return;  // error 처리(custom hook return 값 undefined이기 때문에)
  ...
  }
  ...
}

/* ReviewForm.js */
import useAsync from '../hooks/useAsync';
...
function ReviewForm({ initialValues=INITIAL_VALUES, initialPreview, onCancel, onSubmit, onSubmitSuccess }) {
  const [isSubmitting, submittingError, onSubmitAsync] = useAsync(onSubmit);
  ...
  const handleSubmit = async (e) => {
    ...
    const result = await onSubmitAsync(formData); // CH 19. Custom hook
    if (!result) return;   // error 처리(custom hook return 값 undefined이기 때문에)
    ...
  }
}
```

### Step 4-7. useCallback

```react
/* App.js */
import { useCallback, useEffect, useState } from 'react';
import useAsync from '../hooks/useAsync';
...
function App() {
  ...
  const [isLoading, loadingError, getReviewsAsync] = useAsync(getReviews); 
  ...
  const handleLoad = useCallback(async (options) => {
    const result = await getReviewsAsync(options);
    if (!result) return;
    const { reviews, paging } = result;  
    if (options.offset === 0) {
      setItems(reviews);
    } else {
      setItems((prevItems) => [...prevItems, ...reviews]);   
    }
    setOffset(options.offset + reviews.length);
    setHasNext(paging.hasNext);          
  }, [getReviewsAsync]);

  useEffect(() => {
    handleLoad({ order, offset:0, limit:LIMIT });
  }, [order, handleLoad]);
  ...
}

/* useAsync.js */
import { useCallback, useState } from "react";
...
function useAsync(asyncFunction) {
  ...
  const wrappedFunction = useCallback(async (...args) => {
        try {
            setError(null);
            setPending(true);
            return await asyncFunction(...args);
        } catch (error) {
            setError(error);
            return;
        } finally {
            setPending(false);
        }
    }, [asyncFunction]);
  ...
}

// asyncFunction에 해당하는 것은 useAsync(getReviews);에서 getReviews가 되는데
// 해당 함수는 내부에서 별도로 선언하는 함수가 없으므로 useCallback 미적용
```


<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 5. 전역 데이터 다루기

### Step 5-1. Context

* Context : Context는 프롭 드릴링을 해결하기 위해 사용하는 기능
  

|Prop Drilling|Context|
|---|---|
|[image](1:55)|2:45|
|1:47|2:58|

* Context 만들기

```react
import { createContext } from 'react';
const LocaleContext = createContext();

import { createContext } from 'react';
const LocaleContext = createContext('ko');
```

* Context 적용하기 : 반드시 값을 공유할 범위를 정하고 써야 하는데요,이때 범위는 Context 객체에 있는 Provider 라는 컴포넌트로 정해줌(이때 Provider의 value prop으로 공유할 값을 내려주면 됨)

  ```react
  import { createContext } from 'react';

  const LocaleContext = createContext('ko');

  function App() {
    return (
      <div>
        ... 바깥의 컴포넌트에서는 LocaleContext 사용불가

        <LocaleContext.Provider value="en">
            ... Provider 안의 컴포넌트에서는 LocaleContext 사용가능
        </LocaleContext.Provider>
      </div>
    );
  }
  ```

* Context 값 사용하기(`useContext`)

  ```react
  import { createContext, useContext } from 'react';

  const LocaleContext = createContext('ko');

  function Board() {
    const locale = useContext(LocaleContext);
    return <div>언어: {locale}</div>;
  }

  function App() {
    return (
      <div>
        <LocaleContext.Provider value="en">
            <Board />
        </LocaleContext.Provider>
      </div>
    );
  }
  ```

* State, Hook와 함께 활용하기 : Provider 역할을 하는 컴포넌트를 하나 만들고, 여기서 State를 만들어서 value 로 넘겨줄 수 있습니다. 그리고 아래의 useLocale 같이 useContext 를 사용해서 값을 가져오는 커스텀 Hook을 만들 수도 있겠죠. 이렇게 하면 Context에서 사용하는 State 값은 반드시 우리가 만든 함수를 통해서만 쓸 수 있기 때문에 안전한 코드를 작성하는데 도움이 됩니다.

```react
import { createContext, useContext, useState } from 'react';

const LocaleContext = createContext({});

export function LocaleProvider({ children }) {
  const [locale, setLocale] = useState();
  return (
    <LocaleContext.Provider value={{ locale, setLocale }}>
      {children}
    </LocaleContext.Provider>
  );
}
```

### Step 5-2. 


### Step 5-3. 

<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 6. react 데이터 상태 관리

* Recoil
