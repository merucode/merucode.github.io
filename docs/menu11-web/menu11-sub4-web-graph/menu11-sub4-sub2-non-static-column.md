---
layout: default
title: Non Static Columns Data Graph
parent: Web Graph
grand_parent: Web
nav_order: 2
---

# Non Static Columns Data Graph

{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

<!------------------------------------ STEP ------------------------------------>

## STEP 0. Reference Site

* [Github]()
* [그래프 색상표](https://colorhunt.co/)

## STEP 1. Create Test Data

### Step 1-0. About Data

* Non Static Columns → Handling using JSON type database
* Example Data

	|date|words_count|code|
	|---|---|---|
	|2023-06-01|{"col1": 3, "col2": 5, "col3": 20}|000001|
	|2023-06-02|{"col1": 5, "col2": 7}|000001|
	|2023-06-12|{"col2", 5, "col3": 10}|000001|
	|2023-06-20|{"col4": 13}|000001|
	|2023-06-07|{"col1": 15, "col3": 25}|000002|
	|2023-06-08|{"col6": 7, "col1": 5}|000002|
	|2023-06-12|{"col7": 5}|000002|



### Step 1-1. Setting
* Start From [fastapi-react-postgresql-nginx basic form](https://github.com/merucode/form/tree/fastapi-react-postgresql-nginx_basic)
* Setting ENV files : `.backend.env`, `.database.env`, `frontend/.env`


### Step 1-2. Create Test Table

* `bash`

	```bash
	$ docker compose up -d --build
	$ docker exec -it database /bin/bash
	> su - postgres
	> psql -U test_user -d test_db;
	> DROP TABLE test_table;
	> CREATE TABLE test_table (
		date DATE NOT NULL, 
		words_count JSON NOT NULL,
	    code VARCHAR NOT NULL
	);
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-01','{"col1": 3, "col2": 5, "col3": 20}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-02', '{"col1": 5, "col2": 7}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-12','{"col2": 5, "col3": 10}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-20','{"col4": 13}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-07','{"col1": 15, "col3": 25}', '000002');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-08','{"col6": 7, "col1": 5}', '000002');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-12','{"col7": 5}', '000002');
	```

<br>


## STEP 2. Backend

### Step 2-1. File Structure

* File Structure

	```bash
	📁backend
	├── 📄Dockerfile
	├── 📄database.py
	├── 📁domain
	│   └── 📁words_count
	│       ├── 📄words_count_crud.py
	│       ├── 📄words_count_router.py
	│       └── 📄words_count_util.py
	├── 📄main.py
	├── 📄models.py
	└── 📄requirements.txt
	```

### Step 2-2. Code

* `models.py`

	```python
	from sqlalchemy import Column, String, JSON, Date
	from database import Base

	class WordsCount(Base):
	    __tablename__ = "test_table" 

	    date = Column(Date, nullable=False, primary_key=True)
	    words_count = Column(JSON, nullable=False)
	    code = Column(String, nullable=False)
	```


* `main.py`

	```python
	from fastapi import FastAPI
	from domain.words_count import words_count_router

	app = FastAPI(root_path="/api")

	@app.get("/")
	async def root():
	    return {"message":"Hello"}

	app.include_router(words_count_router.router)
	``` 


* `domain/words_count/words_count_router.py`

	```python
	from typing import Union
	from fastapi import APIRouter, Depends

	from database import get_async_db
	from sqlalchemy.ext.asyncio import AsyncSession 

	from domain.words_count import words_count_crud

	router = APIRouter(
	    prefix="/words-count",
	)

	@router.get("/")
	async def words_count(db:AsyncSession=Depends(get_async_db), 
	        stockCode:Union[str,None]=None, 
	        startDate:Union[str,None]=None, 
	        stopDate:Union[str,None]=None,
	        reqCount:Union[int,None]=None,
	        ):
	    result = await words_count_crud.get_async_words_count(db, stockCode=stockCode, startDate=startDate, stopDate=stopDate, reqCount=reqCount)
	    return result
	``` 

* `domain/words_count/words_count_curd.py`
 
	```python
	from sqlalchemy.ext.asyncio import AsyncSession
	from sqlalchemy import select
	from fastapi.responses import JSONResponse

	from models import WordsCount
	from domain.words_count.words_count_util import common_words_count, make_response_data, fill_null_data

	async def get_async_words_count(db: AsyncSession, stockCode, startDate, stopDate, reqCount):
	    data = await db.execute(select(WordsCount).
	            filter(WordsCount.code==stockCode).
	            filter(WordsCount.date.between(startDate, stopDate)))
	    load_data = data.scalars().fetchall()   # Load data from PG
	    
	    
	    # Extract commom word and count as much as reqCount
	    common_words_count_lst = common_words_count(load_data, reqCount)
	    
		# Make response list of dict fom json data only containing common words 
	    response_dict_lst, db_date_lst  = make_response_data(load_data, common_words_count_lst)
	    
	    # Fill data of null date
	    response_dict_lst = fill_null_data(response_dict_lst, db_date_lst, common_words_count_lst, startDate, stopDate)
	    
	    return JSONResponse(content={"data":response_dict_lst, "comWords":common_words_count_lst})
	```

*  `domain/words_count/words_count_util.py`

	```python
	import functools, operator, collections
	from datetime import datetime, timedelta

	### Extract commom word and count as much as reqCount
	# To access commom word : common_words_count_lst[0][0], common_words_count_lst[1][0] ...
	# To access common word count : common_words_count_lst[0][1], common_words_count_lst[1][1] ...
	def common_words_count(result_data_from_sql, reqCount) -> list:
	    words_dict_list = []
	    for row in result_data_from_sql:
	        row_words_count_dict = dict(row.words_count)
	        words_dict_list.append(row_words_count_dict)
	    
	    # Add dict with sum value if same key
	    result_dict = dict(functools.reduce(operator.add,
	        map(collections.Counter, words_dict_list)))
	   
	    # Sort by value
	    result_sorted_dict = sorted(result_dict.items(), key=lambda item: item[1], reverse = True) 
	    
	    # Handling case when reqCount is bigger than common word
	    if len(result_sorted_dict) < reqCount:
	        reqCount = len(result_sorted_dict)
	        
	    return result_sorted_dict[:reqCount]
	    
	    
	### Make response list of dict fom json data only containing common words 
	# response_dict_lst : list of dictionary that have key of date, common words 
	# db_date_lst       : date which exist in database
	def make_response_data(result_data_from_sql, common_words_count_lst):
	    response_dict_lst = []
	    db_date_lst = []
	    for row in result_data_from_sql:
	        date = row.date
	        db_date_lst.append(datetime.strptime(date, "%Y-%m-%d").date())
	        word_dict = dict(row.words_count)
	        
	        data_dict = {"date":date}
	        for i, (word, count) in enumerate(common_words_count_lst):
	            try:
	                data_dict[common_words_count_lst[i][0]] = word_dict[word] 
	            except:
	                data_dict[common_words_count_lst[i][0]] = 0
	                
	        response_dict_lst.append(data_dict)
	    return response_dict_lst, db_date_lst

	### Fill data of null date
	def fill_null_data(response_dict_lst, db_date_lst, common_words_count_lst, startdate, stopdate):
	    startdate = datetime.strptime(startDate, "%Y-%m-%d").date()
	    stopdate = datetime.strptime(stopDate, "%Y-%m-%d").date()
	    response_dict_lst_total = []
	    k = 0   # for count append response_dict_lst to response_dict_lst_total
	    while startdate <= stopdate:
	        if startdate not in db_date_lst:
	            replace_data = {"date":datetime.strftime(startdate, "%Y-%m-%d")}
	            for i in range(0,len(common_words_count_lst)):
	                replace_data[common_words_count_lst[i][0]] = 0
	            response_dict_lst_total.append(replace_data)        
	        else:   # Already exsist data
	            response_dict_lst_total.append(response_dict_lst[k])
	            k += 1
	        startdate = startdate + timedelta(days=1)
	    return response_dict_lst_total
	``` 

* connect to `[EC2 Public IP]/api/docs` and check

<br>

## STEP 3. Frontend

### Step 3-1. File Structure

* File Structure

	```bash
	📁frontend
	├── 📄Dockerfile
	├── 📁public
	│   └── 📄index.html
	└── 📁src
	    ├── 📄Main.js
	    ├── 📄api.js
	    ├── 📁components
	    │   ├── 📄App.js
	    │   ├── 📄Graph.jsx
	    │   ├── 📄GraphDataList.jsx
	    │   ├── 📄GraphSearchForm.jsx
	    │   └── 📄Header.jsx
	    ├── 📁hooks
	    │   └── 📄useAsync.jsx
	    ├── 📄index.js
	    ├── 📁pages
	    │   ├── 📁GraphPage
	    │   │   └── 📄GraphPage.jsx
	    │   └── 📁HomePage
	    │       └── 📄HomePage.jsx
	    └── 📄urls.js
	```

### Step 3-2. Code

* `bash`

	```bash
	$ docker compose up -d --build
	$ docker exec -it frontend /bin/sh
	> npm install recharts --save
	```

* `Main.js`

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

* `urls.js`

	```jsx
	export const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
	export const BACKEND_GRAPH_URL = BACKEND_URL + "words-count/";
	```

* `api.js`

	```jsx
	import { BACKEND_GRAPH_URL } from "./urls";
	import axios from "axios";

	export async function getItems({ stockCode, startDate, stopDate, reqCount }) {
		const req_config = {
			headers: {
				"Content-type": "application/json",
			},
		};

		const response = await axios.get(
			BACKEND_GRAPH_URL,
			{params: {
				stockCode: stockCode,
				startDate: startDate,
				stopDate: stopDate,
				reqCount: reqCount,
				}
			},
			req_config
		);

		return response.data;
	}
	```


* `pages/HomePage/HomePage.jsx`

	```jsx
	function HomePage() {
	  return (<div><h1>HomePage</h1></div>)
	}
	export default HomePage;
	```

*  `pages/HomePage/GraphPage.jsx`

	```jsx
	import { useState } from 'react';

	import GraphSearchForm from '../../components/GraphSearchForm';
	import GraphDataList from '../../components/GraphDataList';
	import GraphLine from '../../components/GraphLine';

	function GraphPage() {
		const [items, setItems] = useState([]);
		const [comWords,setComWords] = useState([]);

		const handleSubmitSuccess = (res) => {
			setItems(res.data);
			setComWords(res.comWords);
		};

		return (
		<div>
		<h1>GraphPage!</h1>
			<GraphSearchForm onSubmitSuccess={handleSubmitSuccess} />
			<GraphLine items={items} comWords={comWords} />
			<GraphDataList items={items} comWords={comWords} />
		</div>
		)
	}

	export default GraphPage;
	```

* `components/Header.jsx`

	```jsx
	import { Link } from 'react-router-dom';

	function Header() {
	  return (
	  <header>
	    <div>
	      <Link to="/">Web Link</Link>
	      <Link to="/graph">Graph</Link>
	    </div>
	  </header>
	  );
	}

	export default Header;
	```

* `components/GraphSearchForm.jsx` 

	```jsx
	import { useEffect, useState, useCallback } from 'react';

	import { getItems } from '../api';
	import useAsync from '../hooks/useAsync';

	function GraphSearchForm({ onSubmitSuccess }) {
		const [stockCode, setStockCode] = useState("000001");
		const [startDate, setStartDate] = useState("2023-06-01");
		const [stopDate, setStopDate] = useState("2023-06-20");
		const [reqCount, setReqCount] = useState(3);

		const [isLoading, loadingError, getItemsAsync] = useAsync(getItems);

		const loadItems = useCallback(async(options) => {
		    const result = await getItemsAsync(options);
			return result;
		}, [getItemsAsync]);

		const submitHandler = async (e) => {
			e.preventDefault();
			const result = await loadItems({ stockCode, startDate, stopDate, reqCount });
			if (!result) return;
			onSubmitSuccess(result);
		};

		return (
		<div>
			<form onSubmit={submitHandler}>
				<div>
					<label htmlFor="stockCode">stockcode</label>
					<input type="text" id="stockCode" name="stockCode"
					value={stockCode} onChange={(e) => setStockCode(e.target.value)}/>
				</div>
				<div>
					<label htmlFor="startDate">startdate</label>
					<input type="date" id="startDate" name="startDate"
					value={startDate} onChange={(e) => setStartDate(e.target.value)}/>
				</div>
				<div>
					<label htmlFor="stopDate">stopdate</label>
					<input type="date" id="stopDate" name="stopDate"
					value={stopDate} onChange={(e) => setStopDate(e.target.value)}/>
				</div>
				<div>
					<label htmlFor="reqCount">reqCount</label>
					<input type="number" id="reqCount" name="reqCount"
					value={reqCount} onChange={(e) => setReqCount(e.target.value)}/>
				</div>
				<button type="submit">Search</button>
			</form>
			{isLoading && <span>Loading</span>}
			{loadingError?.message && <span>{loadingError.message}</span>}
		</div>
		);
		
	}

	export default GraphSearchForm;
	```

* `components/GraphDataList.jsx`

	```jsx
	function DataListItem({ item, comWords }) {
	  return (
	    <div>
	        <p>{item.date}, {item[comWords[0][0]]}, {item[comWords[1][0]]}, {item[comWords[2][0]
	]}</p>
	    </div>
	  );
	}

	function GraphDataList({ items, comWords }) {
	  return (
	    <>
	      <ul>
	        {items.map((item) => {
	          return (
	            <li key={item.date}>
	              <DataListItem item={item} comWords={comWords}/>
	            </li>
	          );
	        })}
	      </ul>

	      <ul>
	        {comWords.map((comWord) => {
	          return (
	            <li key={comWord[0]}>comWord : {comWord[0]}, count : {comWord[1]}</li>
	          );
	        })}
	      </ul>
	    </>
	  );
	}
	```

* `components/GraphLine.jsx`

	```jsx
	import React, { PureComponent } from 'react';
	import { useSearchParams } from 'react-router-dom';
	import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContaine
	r } from 'recharts';

	function GraphLine({ items, comWords }) {
	  const color = ["#FF002A", "#2600FF", "#03A762"]
	  const lineGraphs = comWords.map((comWord, i) => (
	    <Line type="monotone" dataKey={comWord[0]} stroke={color[i]} activeDot={{ r: 8 }} />
	  ));

	  return (
	      <>
	      {comWords.map((comWord) => {console.log(comWord[0])})}
	      <LineChart
	        width={500}
	        height={300}
	        data={items}
	        margin={{
	          top: 5,
	          right: 30,
	          left: 20,
	          bottom: 5,
	        }}
	      >
	        <CartesianGrid strokeDasharray="3 3" />
	        <XAxis dataKey="date" />
	        <YAxis />
	        <Tooltip />
	        <Legend />
	          {lineGraphs}
	      </LineChart>
	      </>
	  );
	}

	export default GraphLine;
	```