---
layout: default
title: React Router Practice
parent: React
grand_parent: Frontend
nav_order: 6
---

# React Router Practice
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

## STEP 1. React Router Basic Form(with Docker)

### Step 1-1. File Structure

* **File Structure**

	```bash
	📁frontend
	├── 📄Dockerfile
	├── 📁node_modules
	├── 📄package-lock.json
	├── 📄package.json
	├── 📁public
	└── 📁src
		├── 📄Main.js
		├── 📄index.js
		├── 📁components
		│   ├── 📄App.js
		│   └── 📄Header.jsx
		└── 📁pages
		    └── 📄HomePage
	```

### Step 1-2. Code

* `bash`
	*	Install `axios`, `react-router-dom@6`
	```bash
	$ docker compose up -d --build
	$ docker exec -it frontend /bin/sh
	# npm install react-router-dom@6 --save
	```

* `index.js`

	```jsx
	import React from 'react';
	import ReactDOM from 'react-dom/client';
	import Main from './Main';

	const root = ReactDOM.createRoot(document.getElementById('root'));
	root.render(
	  <React.StrictMode>
	    <Main />
	  </React.StrictMode>
	);
	```

* `Main.js`

	```jsx
	import { BrowserRouter, Routes, Route } from 'react-router-dom';
	import App from "./components/App";
	import HomePage from "./pages/HomePage/HomePage";

	function Main() {
	  return (
	    <BrowserRouter>
	      <Routes>
	        <Route path="/" element={ <App /> }>
	          <Route index element={ <HomePage />} />
	        </Route>
	      </Routes>
	    </BrowserRouter>
	  );
	}
	```

* `components/App.js`

	```jsx
	import { Outlet } from 'react-router-dom';
	import Header from '../components/Header';

	function App() {
	  return (
	    <>
	      <Header />
	      <div><Outlet /></div>
	    </>
	  );
	}

	export default App;
	```

* `components/Header.js`

	```jsx
	import React from "react";
	import { Link } from 'react-router-dom';

	function Header() {
	  return (
	  <header>
	    <div>
	      <Link to="/">Web Link</Link>
	    </div>
	  </header>
	  );
	}

	export default Header;
	```

* `pages/HomePage/HomePage.jsx`

	```jsx
	function HomePage() {
	  return (<div>
	    <h1>HomePage</h1>
	  </div>
	  )}

	export default HomePage;
	```