---
layout: default
title: React Axios
parent: React
grand_parent: Frontend
nav_order: 11
---

# React Axios
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

* []()


<br>

## STEP 1. Backend로 Get 요청

### Step 1-1. 기본 사용

* `src/pages/ExPage/ExPage.jsx`

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

* `/src/Main.js`

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

### Step 1-2. useAsync 사용

* `api.js`

	```jsx
	import { WORDSCOUNT_ENDPOINT } from "./constants/urls";

	import axios from "axios";

	export async function getItems({
			stockname,
			startdate,
			stopdate,
		}) {
			const req_config = {
			headers: {
				"Content-type": "application/json",
				},
	} 

	const response = await axios.get(
		WORDSCOUNT_ENDPOINT,
		{params: {
			stockname: stockname,
			startdate: startdate,
			stopdate: stopdate,
			}
		},
		req_config
	)

	return response.data;
	}
	```

* `hooks/useAsync.js`

	```jsx
	import { useCallback, useState } from 'react';

	function useAsync(asyncFunction) {
		const [pending, setPending] = useState(false);
		const [error, setError] = useState(null);

		const wrappedFunction = useCallback(
			async (...args) => {
			setPending(true);
			setError(null);
			try {
				return await asyncFunction(...args);
			} catch (error) {
				setError(error);
				console.log(error);
			} finally {
				setPending(false);
			}
			},
			[asyncFunction]
		);

		return [pending, error, wrappedFunction];
	}

	export default useAsync;
	```

* `pages/GraphPage/GraphPage.js`

	```jsx
	import React, { useState } from 'react';

	import GraphSearchForm from '../../components/GraphSearchForm';
	import GraphDataList from '../../components/GraphDataList';

	function GraphPage() {
		const [items, setItems] = useState([]);

		const handleSubmitSuccess = (res) => {
			setItems(res);
		};

		return (
		<div>
			<GraphSearchForm onSubmitSuccess={handleSubmitSuccess} />
			<GraphDataList items={items} />
		</div>
	)}
	```

* `./components/GraphSearchForm.jsx`

	```jsx
	import React, { useEffect, useState, useCallback } from 'react';

	import { getItems } from '../api';
	import useAsync from '../hooks/useAsync';

	function GraphSearchForm({ onSubmitSuccess }) {
		const [stockname, setStockname] = useState("060380");
		const [startdate, setStartdate] = useState("2020-04-01");
		const [stopdate, setStopdate] = useState("2023-06-20");

		const [isLoading, loadingError, getItemsAsync] = useAsync(getItems);

		const loadItems = useCallback(
			async(options) => {
			const result = await getItemsAsync(options);
			return result;
		}, [getItemsAsync]);

		const submitHandler = async (e) => {
			e.preventDefault();
			const result = await loadItems({ stockname, startdate, stopdate });
			if (!result) return;
			onSubmitSuccess(result);
		};

		return (
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
			{loadingError?.message && <span>{loadingError.message}</span>}
		</div>
		);
	}
	```

* `./components/GraphDataList.jsx`

	```jsx
	function DataListItem({ item }) {
	return (
		<div>
		<p>{item.date}</p>
		<p>{item.word_counts}</p>
		</div>
	);
	}

	function GraphDataList({ items }) {
	return (
		<ul>
		{items.map((item) => {
			return (
			<li key={item.date}>
				<DataListItem item={item} />
			</li>
			);
		})}
		</ul>
	);
	}

	export default GraphDataList;
	```