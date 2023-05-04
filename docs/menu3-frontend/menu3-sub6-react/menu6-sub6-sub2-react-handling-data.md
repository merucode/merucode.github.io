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


* `ReviewList.js`

  ```react
  import './ReviewList.css';  // design이므로 없어도 됨

  function formatDate(value) {
    const date = new Date(value);
    return `${date.getFullYear()}. ${date.getMonth() + 1}. ${date.getDate()}`;
  }

  function ReviewListItem({ item }) {
    return (
      <div className="ReviewListItem">
        <img className="ReviewListItem-img" src={item.imgUrl} alt={item.title} />
        <div>
          <h1>{item.title}</h1>
          <p>{item.rating}</p>
          <p>{formatDate(item.createdAt)}</p>
          <p>{item.content}</p>
        </div>
      </div>
    );
  }

  function ReviewList({ items }) {
    return (
      <ul>
        {items.map((item) => {    // 배열 map 랜더링
          return (
            <li>
              <ReviewListItem item={item} />
            </li>
          );
        })}
      </ul>
    );
  }

  export default ReviewList;
  ```




<br> 

<!------------------------------------ STEP ------------------------------------>
## STEP 4. 

<br>

<!------------------------------------ STEP ------------------------------------>
