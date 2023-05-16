---
layout: default
title: React Router
parent: React
grand_parent: Frontend
nav_order: 3
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

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. Basic use

### Step 2-1. Import

  ```react
  import { BrowserRouter, Routes, Route } from 'react-router-dom';
  ```

### Step 2-2. Divide Pages into Routes

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

### Step 2-3. Move using Link

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

### Step 2-4. NavLink

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

### Step 2-5. Separate Subpages


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

### Step 2-6. useParams

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

### Step 2-7. Handle NotFoundPage

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

### Step 2-8. Redirect(`Navigate`)

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

### Step 2-9. useSearchParams

```react
import { useSearchParams } from 'react-router-dom'; /
...
function CourseListPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initKeyword = searchParams.get('keyword');
  const [keyword, setKeyword] = useState(initKeyword || '');  // keyword가 없는 경우 빈 문자열
  const courses = getCourses(initKeyword);
  ..
  const handleSubmit = (e) => {
  e.preventDefault();
  setSearchParam(keyword ? { keyword } : {}); // 문자열 아닌 경우 빈 객체 처리
  };
  ...
  return(
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




