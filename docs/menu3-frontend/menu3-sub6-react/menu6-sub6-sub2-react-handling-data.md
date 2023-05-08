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

## STEP 1. 

### Step 1-1. mock 데이터 추가하기

- mock 데이터 : 네트워크에서 받아올 데이터를 흉내낸 데이터

* `App.js`

  ```react
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
  ```

* `ReviewList.js`

  ```react
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
## STEP 4. 

<br>

<!------------------------------------ STEP ------------------------------------>
