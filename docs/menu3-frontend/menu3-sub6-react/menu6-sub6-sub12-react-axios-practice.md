---
layout: default
title: React Axios Practice
parent: React
grand_parent: Frontend
nav_order: 11
---

# React Axios Practice
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

## STEP 0. Related Site

* [https://wonit.tistory.com/305](https://wonit.tistory.com/305)
* [https://velog.io/@mudidu/React-axios-%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC-%EC%84%9C%EB%B2%84%EC%97%90-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9A%94%EC%B2%AD%ED%95%98%EA%B8%B0](https://velog.io/@mudidu/React-axios-%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC-%EC%84%9C%EB%B2%84%EC%97%90-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9A%94%EC%B2%AD%ED%95%98%EA%B8%B0)



<br>

## STEP 1. Backend로 Get 요청

### Step 1-1. `src/pages/ExPage/ExPage.jsx`

```jsx
import React, { useEffect, useState } from 'react';	  
import axios from "axios";

import { BACKEND_ENDPOINT } from "../../constants/urls";

function GraphPage() {
const [stockname, setStockname] = useState("");
const [startdate, setStartdate] = useState("2023-04-01");
const [stopdate, setStopdate] = useState("2023-06-20");
  
const [loading, setLoading] = useState(false);
const [results, setResults] = useState(false);
const [success, setSuccess] = useState(false);

useEffect(() => {
	if (loading) {
		console.log(WORDSCOUNT_ENDPOINT);
		console.log("Calling axios.");
		const req_config = {
		headers: {
		"Content-type": "application/json",
		},
		};
	axios.get(
		WORDSCOUNT_ENDPOINT, 
		{params: {
				stockname: stockname,
				startdate: startdate,
				stopdate: stopdate,
				}
		},
		req_config
		)
		.then((response) => {
			console.log("Axios successful.");
			setResults(true);
			setLoading(false);
			setSuccess(true);
		})
		.catch((error) => {
			console.log("Axios failed.");
			setResults(true);
			setLoading(false);
			setSuccess(false);
		});
		}
}, [loading]);

const submitHandler = (e) => {
	e.preventDefault();
	setLoading(true);
	setResults(false);
	setSuccess(false);
};

return (
	<div>
		<h1>GraphPage!</h1>
		<div>
			<form onSubmit={submitHandler}>
			<div>
				<label htmlFor="stockname">stockname</label>
				<input type="text" id="stockname" name="stockname" 
				value={stockname} onChange={(e) => setStockname(e.target.value)}/>
			</div>
			<div>
				<label htmlFor="startdate">startdate</label>
				<input type="date" id="startdate" name="startdate" 
				value={startdate} onChange={(e) => setStartdate(e.target.value)}/>
			</div>
			<div>
				<label htmlFor="stopdate">stopdate</label>
				<input type="date" id="stopdate" name="stopdate"
				value={stopdate} onChange={(e) => setStopdate(e.target.value)}/>
			</div>
				<button type="submit">Register</button>
			</form>
		</div>
	</div>
)}

export default GraphPage;
```


## Step 1-2.  `/src/Main.js`

```jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import App from "./components/App";
import HomePage from "./pages/HomePage/HomePage";
import GraphPage from "./pages/GraphPage/GraphPage";

function Main() {
  return (
    <BrowserRouter>
		<Routes>
			<Route path="/" element={ <App /> }>
				<Route index element={ <HomePage />} />
				<Route path="/graph" element={ <GraphPage />} />
			</Route>
		</Routes>
    </BrowserRouter>
  );
}
export default Main;
```