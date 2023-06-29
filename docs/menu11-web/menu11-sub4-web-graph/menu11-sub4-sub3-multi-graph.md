---
layout: default
title: Multi Graph
parent: Web Graph
grand_parent: Web
nav_order: 3
---

# Multi Graph

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

* [Github](https://github.com/merucode/fastapi-react-nginx/tree/11_non_static_col_data_graph)
	* Setting ENV files : `.backend.env`, `frontend/.env`
	* [Create test table](https://merucode.github.io/docs/menu11-web/menu11-sub4-web-graph/menu11-sub4-sub2-non-static-column.html#step-1-2-create-test-table)
* [그래프 색상표](https://colorhunt.co/)

## STEP 1. Create Test Data

### Step 1-1. Setting
* Start From [Non Static Columns Data Graph](https://merucode.github.io/docs/menu11-web/menu11-sub4-web-graph/menu11-sub4-sub2-non-static-column.html)
* Setting ENV files : `.backend.env`, `frontend/.env`


### Step 1-2. Create Test Table

* `bash`

	```bash
	$ docker compose up -d --build
	$ docker exec -it database /bin/bash
	> su - postgres
	> psql -U test_user -d test_db;
	> DROP TABLE test_table;
	> CREATE TABLE test_table (
		date VARCHAR(30) NOT NULL, 
		words_count JSON NOT NULL,
	    code VARCHAR(10) NOT NULL
	);
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-01','{"col1": 3, "col2": 5, "col3": 20}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-02', '{"col1": 5, "col2": 7}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-04','{"col2": 5, "col3": 10}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-05','{"col4": 13}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-01','{"col1": 15, "col3": 25}', '000002');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-03','{"col6": 7, "col1": 5}', '000002');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-04','{"col7": 5}', '000002');
	
	
	> CREATE TABLE stock_table (
		date VARCHAR(30) NOT NULL, 
		open INTEGER NOT NULL,
		high INTEGER NOT NULL,
		low INTEGER NOT NULL,
		close INTEGER NOT NULL,
		volume BIGINT NOT NULL,
	    code VARCHAR(10) NOT NULL
	);
	
	INSERT INTO stock_table (date, open, high, low, close, volume, code) VALUES ('2023-05-31', 1000, 1200,  900,  1100, 110005, '000001');
	INSERT INTO stock_table (date, open, high, low, close, volume, code) VALUES ('2023-06-01', 900, 1000,  900,  1000, 170005, '000001');
	INSERT INTO stock_table (date, open, high, low, close, volume, code) VALUES ('2023-06-02', 1000, 1500, 1000,  1500, 270005, '000001');
	INSERT INTO stock_table (date, open, high, low, close, volume, code) VALUES ('2023-06-05', 1500, 1800, 1400,  1700, 870005, '000001');
	
	INSERT INTO stock_table (date, open, high, low, close, volume, code) VALUES ('2023-05-31', 800, 1000, 700, 700, 110005, '000002');
	INSERT INTO stock_table (date, open, high, low, close, volume, code) VALUES ('2023-06-01',700, 700, 550, 600, 170005, '000002');
	INSERT INTO stock_table (date, open, high, low, close, volume, code) VALUES ('2023-06-02', 600, 650, 400, 500, 270005, '000002');
	INSERT INTO stock_table (date, open, high, low, close, volume, code) VALUES ('2023-06-05', 500, 600, 350, 400, 170005, '000002');
	
	
	> CREATE TABLE index_table (
		date VARCHAR(30) NOT NULL, 
		open REAL NOT NULL,
		high REAL NOT NULL,
		low REAL NOT NULL,
		close REAL NOT NULL,
		volume BIGINT NOT NULL,
	    code VARCHAR(10) NOT NULL
	);
	
	INSERT INTO index_table (date, open, high, low, close, volume, code) VALUES ('2023-05-31', 2000.71, 2010.78, 1980.15, 1960.01, 200000, 'KOSPI');
	INSERT INTO index_table (date, open, high, low, close, volume, code) VALUES ('2023-06-01', 1972.12, 1980.22, 1955.2, 1960.12, 190000, 'KOSPI');
	INSERT INTO index_table (date, open, high, low, close, volume, code) VALUES ('2023-06-02', 1960.04, 1962.1, 1900.2, 1910.8, 140000, 'KOSPI');
	INSERT INTO index_table (date, open, high, low, close, volume, code) VALUES ('2023-06-05', 1910.85, 1920.1, 1801.1, 1810.8, 100000, 'KOSPI');
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
	│   └── 📁words_count
	│       ├── 📄stock_index_util.py
	│       ├── 📄words_count_crud.py
	│       ├── 📄words_count_router.py
	│       └── 📄words_count_util.py
	├── 📄main.py
	├── 📄models.py
	└── 📄requirements.txt
	```

### Step 2-2. Code

* `models.py`

	```python
	from sqlalchemy import Column, String, Date, JSON, BIGINT, Integer, REAL
	
	from database import Base
	
	class WordsCount(Base):
	    __tablename__ = "words_count_table"
	
	    date = Column(Date, nullable=False, primary_key=True)
	    words_count = Column(JSON, nullable=False)
	    code = Column(String, nullable=False)
	
	
	class StockOHLCV(Base):
	    __tablename__ = "stock_table"
	
	    date = Column(Date, nullable=False, primary_key=True)
	    open = Column(Integer, nullable=False)
	    high = Column(Integer, nullable=False)
	    low = Column(Integer, nullable=False)
	    close = Column(Integer, nullable=False)
	    volume = Column(BIGINT, nullable=False)
	    code = Column(String, nullable=False)
	
	
	class IndexOHLCV(Base):
	    __tablename__ = "index_table"
	
	    date = Column(Date, nullable=False, primary_key=True)
	    open = Column(REAL, nullable=False)
	    high = Column(REAL, nullable=False)
	    low = Column(REAL, nullable=False)
	    close = Column(REAL, nullable=False)
	    volume = Column(BIGINT, nullable=False)
	    code = Column(String, nullable=False)
	```


* `domain/words_count/words_count_curd.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi.responses import JSONResponse
import pandas as pd

from models import WordsCount, StockOHLCV, IndexOHLCV
from domain.words_count.words_count_util import common_words_count, make_response_data, fill_null_data
from domain.words_count.stock_index_util import convert_to_dataframe, fill_null_data_stock_index, merge_list_df

async def get_async_words_count(db: AsyncSession, stockCode, startDate, stopDate, reqCount):
    data_words_count = await db.execute(select(WordsCount).
            filter(WordsCount.code==stockCode).
            filter(WordsCount.date.between(startDate, stopDate)))
    load_data_words_count = data_words_count.scalars().fetchall()   # Load data from PG
    
    data_stock_ohlcv = await db.execute(select(StockOHLCV).
            filter(StockOHLCV.code==stockCode).
            filter(StockOHLCV.date.between(startDate, stopDate)))
    load_data_stock_ohlcv = data_stock_ohlcv.scalars().fetchall()
    df_stock = convert_to_dataframe(load_data_stock_ohlcv, ['date', 'close'], stockCode)
    
    data_index_ohlcv = await db.execute(select(IndexOHLCV).
            filter(IndexOHLCV.code=="KOSPI").
            filter(IndexOHLCV.date.between(startDate, stopDate)))
    load_data_index_ohlcv = data_index_ohlcv.scalars().fetchall() 
    df_index= convert_to_dataframe(load_data_index_ohlcv, ['date', 'close'], 'KOSPI')
    
    # Merge index and stock data and fill null data in stock index data
    df_stock_index = pd.merge(df_stock, df_index, on='date', how='outer')
    df_stock_index = fill_null_data_stock_index(df_stock_index, startDate, stopDate)
	# Extract commom word and count as much as reqCount
    common_words_count_lst = common_words_count(load_data_words_count, reqCount)

    # Make response list of dict fom json data only containing common words 
    response_dict_lst, db_date_lst  = make_response_data(load_data_words_count, common_words_count_lst)
    
# Fill data of null date in words_count_dict_lst
    response_dict_lst = fill_null_data(response_dict_lst, db_date_lst, common_words_count_
lst, startDate, stopDate)

    # Merge words_count_data and stock_index_data
    response_dict_lst = merge_list_df(response_dict_lst, df_stock_index, [stockCode])  

    return JSONResponse(content={"data":response_dict_lst, "comWords":common_words_count_lst})
```

* `domain/words_count/words_count_util.py`

	```python
	import pandas as pd
	from datetime import datetime, timedelta

	### Conver model object to dataframe
	def convert_to_dataframe(model_data, col_names, code):
		data = [row.__dict__ for row in model_data]
		df = pd.DataFrame(data)
		df = df[col_names]

		# Column Name Custom
		df = df.rename(columns={'close':code})
		return df

	### Fill null data
	def fill_null_data_stock_index(df, startdate, stopdate):
		date_range = pd.date_range(start=startdate, end=stopdate, freq='D')    # Create date range
		# Add startdate, stopdate to DataFrame
		df = pd.concat([pd.DataFrame({'date': [startdate]}), df, pd.DataFrame({'date': [stopdate]})], ignore_index=True)
		df['date'] = pd.to_datetime(df['date'])
		df = df.sort_values('date') # Sort by date

		df = df.groupby('date').first().reindex(date_range).reset_index()   # Resampling as Date
		df = df.rename(columns={'index':'date'})
		df['date'] = df['date'].dt.strftime('%Y-%m-%d')
		df = df.fillna(method='ffill')  # Fill NaN Data using data(pre:ffill, after:bfill)
		return df

	### Merge df and list of dictionary
	def merge_list_df(lst_data, df_data, stockCode) -> list:
		result_dict_lst = []
		codes = [stockCode[0], 'KOSPI']

		for row in lst_data:
			dict = row
			date = dict['date']
			for code in codes:
				dict[code] = df_data[df_data['date']==date][[code]].iat[0,0]
			result_dict_lst.append(dict)
		return result_dict_lst
	```

* connect to `[EC2 Public IP]/api/docs` and check

<br>

여기까지 수행~~!!!!!!!!!!!!!!
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
	    │   ├── 📄GraphLine.jsx
		│   ├── 📄GraphPie.jsx
	    │   ├── 📄GraphDataList.jsx
	    │   ├── 📄GraphSearchForm.jsx
	    │   └── 📄Header.jsx
	    ├── 📁hooks
	    │   └── 📄useAsync.jsx
	    ├── 📄index.js
	    ├── 📁pages
	    │   ├── GraphPage
	    │   │   └── GraphPage.jsx
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

*  `pages/GraphPage/GraphPage.jsx`

	```jsx
	import { useState } from 'react';

	import GraphSearchForm from '../../components/GraphSearchForm';
	import GraphDataList from '../../components/GraphDataList';
	import GraphLine from '../../components/GraphLine';
	import GraphPiefrom'../../components/GraphPie';
	
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
			<GraphPiecomWords={comWords} />
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
			<p>{item.date} {comWords.map((e) => <span>{item[e[0]]} </span>)}</p>   
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

	export default GraphDataList;
	```

* `components/GraphLine.jsx`

	```jsx
	import React, { PureComponent } from 'react';
	import { useSearchParams } from 'react-router-dom';
	import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

	function GraphLine({ items, comWords }) {
	  const colors = ["#FF0060", "#0A6EBD", "#00DFA2", "#884A39", "#FFC26F", "#080202", "#40128B"]

	  // Modify for multi graph
	  const lineGraphs = comWords.map((comWord, i) => (
	    <Line type="monotone" dataKey={comWord[0]} stroke={colors[i]} activeDot={{ r: 8 }} />
	  ));

	  return (
	      <>
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
	          {lineGraphs}	// Modify for multi graph
	      </LineChart>
	      </>
	  );
	}

	export default GraphLine;
	```

* `components/GraphPie.jsx`

	```jsx
	import React, { useCallback, useState } from "react";
	import { PieChart, Pie, Sector, Cell } from "recharts";
	
	// rechart const
	const renderActiveShape = (props: any) => {
	const RADIAN = Math.PI / 180;
	const {
		cx,
		cy,
		midAngle,
		innerRadius,
		outerRadius,
		startAngle,
		endAngle,
		fill,
		payload,
		percent,
		value
	} = props;
	const sin = Math.sin(-RADIAN * midAngle);
	const cos = Math.cos(-RADIAN * midAngle);
	const sx = cx + (outerRadius + 10) * cos;
	const sy = cy + (outerRadius + 10) * sin;
	const mx = cx + (outerRadius + 30) * cos;
	const my = cy + (outerRadius + 30) * sin;
	const ex = mx + (cos >= 0 ? 1 : -1) * 22;
	const ey = my;
	const textAnchor = cos >= 0 ? "start" : "end";
	
	return (
		<g>
		<text x={cx} y={cy} dy={8} textAnchor="middle" fill={fill}>
			{payload.name}
		</text>
		<Sector
			cx={cx}
			cy={cy}
			innerRadius={innerRadius}
			outerRadius={outerRadius}
			startAngle={startAngle}
			endAngle={endAngle}
			fill={fill}
		/>
		<Sector
			cx={cx}
			cy={cy}
			startAngle={startAngle}
			endAngle={endAngle}
			innerRadius={outerRadius + 6}
			outerRadius={outerRadius + 10}
			fill={fill}
		/>
		<path
			d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`}
			stroke={fill}
			fill="none"
		/>
		<circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
		<text
			x={ex + (cos >= 0 ? 1 : -1) * 12}
			y={ey}
			textAnchor={textAnchor}
			fill="#333"
		>{`${value}`}</text>
		<text
			x={ex + (cos >= 0 ? 1 : -1) * 12}
			y={ey}
			dy={18}
			textAnchor={textAnchor}
			fill="#999"
		>
			{`(${(percent * 100).toFixed(2)}%)`}
		</text>
		</g>
	);
	};
	
	function GraphPie ({ comWords }) {
		// Transform data format as array of object for piechart(rechart)
		// format : [{"name":...,"value":...}, {...}, ...]
		const dataKey = ["name", "value"]
		let obj = {};
		const data = comWords.map((e) => {
			e.forEach((e, i) => obj[dataKey[i]] = e);
			return {...obj};
		});
	
		const colors = ["#FF0060", "#0A6EBD", "#00DFA2", "#884A39", "#FFC26F", "#080202", "#40128B"]
	
		//rechart pie-chart-with-customized-active-shape
		const [activeIndex, setActiveIndex] = useState(0);
		const onPieEnter = useCallback(
		(_, index) => {
			setActiveIndex(index);
		},
		[setActiveIndex]
		);
	
		return (
		<PieChart width={400} height={400}>
			<Pie
			activeIndex={activeIndex}
			activeShape={renderActiveShape}
			data={data}
			cx={200}
			cy={200}
			innerRadius={60}
			outerRadius={80}
			fill="#8884d8"
			dataKey="value"
			onMouseEnter={onPieEnter}
			>
			{data.map((entry, index) =>
				<Cell key={index} fill={colors[index]}/>
			)
			}
			</Pie>
		</PieChart>
		);
	};
	
	export default GraphPie;
	```