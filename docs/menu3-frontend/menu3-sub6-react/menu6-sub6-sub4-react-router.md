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
    ```