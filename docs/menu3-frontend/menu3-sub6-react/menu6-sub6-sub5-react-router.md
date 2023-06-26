---
layout: default
title: React Router
parent: React
grand_parent: Frontend
nav_order: 5
---

# React Router
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

## STEP 1. React Router

### Step 1-1. Install Ract Router

  ```bash
  $ npm install react-router-dom@6
  ```
  
  * 예제를 위해선 `npm install classnames` 설치 필요

### Step 1-2. Ract Router V6

  * React Router : The library that divide page into react component
  * Component
    * Router : Internally, it is context provider. So, `Routes` must be used within the `Router`
    * Routes, Route : Similar to switch/case
    * Link : Used instead of a tag    

  ```react
  import { BrowserRouter } from 'react-router-dom';
  import App from './components/App';
  import HomePage from './pages/HomePage';

  function Main() {
    return (
      <BrowserRouter>
        <App>
          <HomePage />
        </App>
      </BrowserRouter>
    );
  }

  export default Main;
  ```

### Step 1-3. CSR, SPA

* 클라이언트사이드 렌더링(CSR) : 웹 브라우저에서 자바스크립트로 HTML 페이지를 만드는 것
* 싱글페이지 어플리케이션(SPA) : 하나의 HTML 문서 안에서 자바스크립트로 여러페이지를 보여주는 사이트
* React Router 사용 시 CSR과 SPA 구현됨

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. Basic use

### Step 2-1. `Router`

* 리액트 라우터를 사용하려면 반드시 라우터라는 컴포넌트가 필요한데요. 우리는 그 중에서도 BrowserRouter 라는 걸 사용했습니다. 이 컴포넌트를 최상위 컴포넌트에서 감싸주면 모든 곳에서 사용할 수 있습니다. 

  ```react
  import { BrowserRouter } from 'react-router-dom';

  function App() {
    return <BrowserRouter> ... </BrowserRouter>;
  }
  ```

### Step 2-2. `Route`(How to divide pages)

* Routes 컴포넌트 안에다가 Route 컴포넌트를 배치해서 각 페이지를 나눠줄 수 있습니다. 이때 Routes 안에서는 위에서부터 차례대로 Route를 검사하는데요. 현재 경로와 path prop이 일치하는 Route 를 찾습니다.

  ```react
  <Routes>
    <Route path="/" element={<HomePage />} />
    <Route path="posts" element={<PostListPage />} />
    <Route path="posts/1" element={<PostPage />} />
  </Routes>
  ```

### Step 2-3. `Link`

  ```react
  <Link to="/posts">블로그</Link>
  ```

### Step 2-4. `Route`(How to divide subpage)

* Route 컴포넌트 안에다가 Route 컴포넌트를 배치하면 됩니다. 이때 하위 페이지에서 최상위 경로에 해당하는 경로는 path prop이 아니라 index 라는 prop을 사용하면 됩니다.

  ```react
  <Routes>
    <Route path="/"><HomePage /></Route>
    <Route path="posts" element={<PostLayout />} >
      <Route index element={<PostListPage />}  />
      <Route path="1" element={<PostPage />}  />
    </Route>
  </Routes>
  ```

* 이때 부모 Route 컴포넌트에 element 를 지정하고, 아래처럼 Outlet 이라는 컴포넌트를 활용하면 공통된 레이아웃을 지정해줄 수 있었죠.

  ```react
  import { Outlet } from 'react-router-dom';

  function PostLayout() {
    return (
      <div>
        <h1>블로그</h1>
        <hr />
        <Outlet />
      </div>
    );
  }

  export default PostLayout;
  ```

### Step 2-5. `useParams`(dynamic path)

* 콜론 (:) 으로 시작하는 문자열을 사용하면 경로에 파라미터를 지정할 수 있었습니다. 예를들어서 아래처럼 /posts/:postId 라는 경로는 /posts/123 이라던지 /posts/abc 라는 주소로 접속하면 123 이나 abc 라는 값을 postId 라는 파라미터로 받습니다.

  ```react
  <Routes>
    <Route path="/"><HomePage /></Route>
    <Route path="posts" element={<PostLayout />} >
      <Route index element={<PostListPage />}  />
      <Route path=":postId" element={<PostPage />}  />
    </Route>
  </Routes>
  ```

* 경로 파라미터를 사용하려면 useParams 라는 훅을 사용하면 됩니다.

  ```react
  function PostPage() {
    const { postId } = useParams();
    // ...
  }
  ```

### Step 2-6. `useSearchParams`(use query)

* useSearchParams 라는 Custom hook으로 SearchParams 객체를 받아올 수 있는데요. 이 hook은 SearchParams 객체와 Setter 함수를 배열형으로 리턴합니다. 이때 쿼리 값은 SearchParams 의 get 함수로 가져옵니다.

  ```react
  import { useSearchParams } from 'react-router-dom';

  function PostListPage() {
    const [searchParams, setSearchParams] = useSearchParams();
    const filterQuery = searchParams.get('filter');

    // ...
  }
  ```

* 만약 쿼리 값을 변경하고 주소를 이동하고 싶다면 Setter 함수에 객체를 넘겨주면 됩니다. 이때 객체의 프로퍼티로 쿼리 값을 지정할 수 있습니다.
아래 예시는 ?filter=react 라는 쿼리로 이동하는 예시입니다.

  ```react
  setSearchParams({
    filter: 'react',
  });
  ```

### Step 2-7. `Navigate`(move page)

* 리턴값으로 Navigate 컴포넌트를 리턴하면 to prop으로 지정한 경로로 이동합니다.

  ```react
  function PostPage() {
    // ...
    const post = getPost(postId);
    // post가 없는 경우 /posts 페이지로 이동
    if (!post) {
      return <Navigate to="/posts" />;
    }
    // ...
  }
  ```

### Step 2-8. `useNavigate`(move page)

* useNavigate 라는 hook으로 navigate 함수를 가져오면 이 함수를 통해 페이지를 이동할 수 있습니다.

  ```react
  const navigate = useNavigate();

  const handleClick = () => {
    // ... 어떤 작업을 한 다음에 페이지를 이동
    navigate('/wishlist');
  }
  ```

### Step 2-9. Link, Navigate, useNavigate

* Link : 사용자가 클릭해서 페이지를 이동하도록 할 때 
* Navigate : 특정 경로에서 렌더링 시점에 다른 페이지로 이동시키고 싶을 때
  * 쇼핑몰의 회원 전용 페이지에 로그인없이 들어와서 로그인 페이지로 리다이렉트하는 경우
  * 쇼핑몰의 상품 상세 페이지에서 제품이 품절되었거나 삭제되어서 다른 페이지로 이동시키는 경우
* useNavigate : 특정한 코드의 실행이 끝나고 나서 페이지를 이동시키고 싶을 때
  * 쇼핑몰에서 결제하기 버튼을 누르고 나서 모든 결제가 완료된 후에 페이지를 이동시키는 경우
  * 쇼핑몰에서 장바구니에 담기를 눌렀을 때 리퀘스트를 보내고 장바구니 페이지로 이동시키는 경우
  * 리다이렉트된 로그인 페이지에서 로그인을 완료한 후에 처음 진입했던 페이지로 돌아가는 경우

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. Example

### Step 3-1. Import

  ```react
  import { BrowserRouter, Routes, Route } from 'react-router-dom';
  ```

### Step 3-2. Divide Pages into Routes

  * Element uses `JSX syntax`

  ```react
  /* Main.js */
  import { BrowserRouter, Routes, Route } from 'react-router-dom';
  import App from './components/App';
  import HomePage from './pages/HomePage';
  import CoursePage from './pages/CoursePage';
  import CourseListPage from './pages/CourseListPage';
  import WishlistPage from './pages/WishlistPage';

  function Main() {
    return (
      <BrowserRouter>
        <App>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="courses" element={<CourseListPage />} />
            <Route
              path="courses/react-frontend-development"
              element={<CoursePage />}
            />
            <Route path="wishlist" element={<WishlistPage />} />
          </Routes>
        </App>
      </BrowserRouter>
    );
  }

  export default Main;
  ```

### Step 3-3. Move using Link

  ```react
  /* Nav.js */
  import { Link } from 'react-router-dom';
  ...
  function Nav() {
    return (
      ...
      <Link to="/">
        <img src={logoImg} alt="Codethat Logo" />
      </Link>
      ...
      <Link to="/courses">카탈로그</Link>
      ...
    );
  }
  export default Nav;

  /* QuestionListPage.js */
  ...
  function QuestionItem({ question }) {
    return (
      ...
      <Link to={`/question/${question.id}`}>{question.title}</Link>
      ...
    );
  }
  ...
  ```

### Step 3-4. NavLink

* NavLink : Style(inline) can be used inside `NavLink` tags

```react
/* Nav.js */
import { Link, NavLink } from 'react-router-dom';
...
function getLinkStyle({ isActive }) {
  return {
    textDecoration: isActive ? 'underline' : undefined,
  }
}

function Nav() {
  return (
    ...
     <NavLink to="/courses" style={getLinkStyle}>카탈로그</NavLink>
    <NavLink to="/questions" style={getLinkStyle}>커뮤니티</NavLink>
    ...
  );
}
```

### Step 3-5. Separate Subpages


  ```react
  <Route path="courses">                  
    <Route index element={<CourseListPage />} />
    <Route path="react-frontend-development" element={<CoursePage />} />
  </Route>
  ```

  ```react
  /* Main.js */
  ...
  function Main() {
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<App />}>
            <Route index element={<HomePage />} />
            <Route path="courses">
              <Route index element={<CourseListPage />} />
              <Route path="react-frontend-development" element={<CoursePage />} />
            </Route>
            <Route path="questions">
              <Route index element={<QuestionListPage />} />
              <Route path="questions/616825" element={<QuestionPage />} />
            </Route>
            <Route path="wishlist" element={<WishlistPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    );
  }

  export default Main;

  /* App.js */  // Outlet에서 렌더링 됨(공통 스타일 적용)
  import { Outlet } from 'react-router-dom';
  import Nav from '../components/Nav';
  import Footer from '../components/Footer';
  import styles from './App.module.css';
  import './App.font.css';

  function App() {
    return (
      <>
        <Nav className={styles.nav} />
        <div className={styles.body}><Outlet /></div>
        <Footer className={styles.footer} />
      </>
    );
  }

  export default App;
  ```

### Step 3-6. useParams

* get request(courseItem에서 지정된 course/뒤 주소) → Route path(courseSlug로 입력) → api getCourseBySlug(courseSlug와 course.slug가 같은 데이터 반환)

  ```react
  /* Main.js */
  ...
  function Main() {
    return (
      ...
      <Route path="courses">
        <Route index element={<CourseListPage />} />
        <Route path=":courseSlug" element={<CoursePage />} />
      </Route>
      ...
    );
  }


  /* components/CourseItem.js */
  ...
  function CourseItem({ course }) {
    return (
      ...
      <Link to={`/courses/${course.slug}`}>{course.title}</Link>
      ...
    );
  }
  export default CourseItem;


  /* page/CourseListPage.js */
  ...
  function CourseListPage() {
    ...
    return (
      {courses.map((course) => (
        <CourseItem key={course.id} course={course} />
      ))}
    );
  }
  export default CourseListPage;


  /* pages/CoursePage */
  import { useParams } from 'react-router-dom';
  import { getCourses } from '../api';
  ...
  function CoursePage() {
    const { courseSlug } = useParams();
    const course = getCourseBySlug(courseSlug);
    ...
  }


  /* api.js */
  ...
  export function getCourseBySlug(courseSlug) {
    return courses.find((course) => course.slug === courseSlug);
  }
  ```

### Step 3-7. Handle NotFoundPage

  ```react
  /* Main.js */
  ...
  function Main() {
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<App />}>
            ...
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    );
  }
  ```

### Step 3-8. Redirect(`Navigate`)

  ```react
  /* pages/CoursePage.js */
  import { Navigate, useParams } from 'react-router-dom'; 
  ...
  function CoursePage() {
    const { courseSlug } = useParams();         // 1. url로 부터 courseSlug(useParams) 받아 와서
    const course = getCourseBySlug(courseSlug); // 2. api를 통해 해당 데이터 조회 후 없다면 course false
    ...
    if (!course) return <Navigate to="/courses" />;         // 3. redirect 처리
    ...
  }
  ```

### Step 3-9. `useSearchParams`

* url 쿼리스트링 `?keyword=react` 입력 시 → react는 자동으로 `{keyword:react}` 객체를 생성 → useSearchParams() 함수로 해당 객체 반환 → `.get('keyword');`로 `keyword` 할당 값 반환 가능  

  ```react
  /* pages/CourseListPage.js */
  import { useSearchParams } from 'react-router-dom'; /
  ...
  function CourseListPage() {
    const [searchParams, setSearchParams] = useSearchParams();
    const initKeyword = searchParams.get('keyword');
    const [keyword, setKeyword] = useState(initKeyword || '');  // keyword가 없는 경우 빈 문자열 처리
    const courses = getCourses(initKeyword);
    ..
    const handleSubmit = (e) => {
      e.preventDefault();
      setSearchParam(keyword ? { keyword } : {}); // 문자열 아닌 경우 빈 객체 처리
    };
    ...
    return(
      ...
      <form className={searchBarStyles.form} onSubmit={handleSubmit}>
        <input name="keyword" value={keyword} onChange={handleKeywordChange} placeholder="검색으로 코스 찾기"
        ></input>
        <button type="submit">
          <img src={searchIcon} alt="검색" />
        </button>
      </form>
      ...
      {initKeyword && courses.length === 0 ? (
        <Warn
            className={styles.emptyList}
            title="조건에 맞는 코스가 없어요."
            description="올바른 검색어가 맞는지 다시 한 번 확인해 주세요."
          />
        ) : (
          <div className={styles.courseList}>
            {courses.map((course) => (
              <CourseItem key={course.id} course={course} />
            ))}
          </div>
        )}
    );
  }
  ```

### Step 3-10. `useNavigate`(page move)

  ```react
  import { Navigate, useParams, useNavigate } from 'react-router-dom'; 
  ...
  function CoursePage() {
    const { courseSlug } = useParams();
    const navigate = useNavigate(); 
    ...
    const handleAddWishlistClick = () => {
      addWishlist(course?.slug);
      navigate('/wishlist');
    };
    ...
  }
  ```

<!------------------------------------ STEP ------------------------------------>

## STEP 4. React Randering

### Step 4-1. Randering Type

* 클라이언트사이드 렌더링(Client-side Rendering) : 자바스크립트로 변환된 리액트 코드를 웹 브라우저에서 실행해서 HTML을 만드는 것

* 서버사이드 렌더링(Server-side Rendering) : 서버에서 HTML을 만들어서 보내줌(렌더링 된 것이 웹 브라우저에 도착하니까 훨씬 빨리 화면을 띄워줄 수 있고,검색 엔진에서 좋은 점수를 받아서 검색했을 때 사이트가 잘 노출될 수 있다는 장점)

* 정적 사이트 생성(Static Site Generation) : 미리 HTML 파일을 만들어서 서버를 배포하는 것

### Step 4-2. Three React Techniques Using Rendering

* Next.js 
  * 리액트 서버사이드 렌더링을 편하게
  * 리액트 라우터랑은 다르게 HTML 파일을 나누듯이 자바스크립트 파일을 나눠 놓으면 곧바로 페이지로 사용할 수 있다는 장점

* Gatsby
  * 리액트로 정적 사이트 만들기
  *  Gatsby는 리액트 코드를 미리 렌더링 해서 프로젝트를 빌드할 때 HTML 파일로 만들어 줌(회사 소개 사이트나 동아리 홈페이지 혹은 포트폴리오 사이트 같이 정적인 사이트)

* React Native
  * 모바일 앱의 화면도 리액트
  * React Native는 이런 아이디어에서 출발한 기술입니다. 리액트로 작성한 코드를 모바일 앱으로 만들 수 있게 해 주는데요. 리액트 코드로 개발하면 웹과 안드로이드와 iOS 앱에서 사용하는 공통적인 코드를 한 번에 개발할 수 있다는 장점
