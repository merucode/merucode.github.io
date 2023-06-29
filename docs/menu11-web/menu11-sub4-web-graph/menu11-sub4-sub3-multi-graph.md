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
	* [Create test table]
	* 
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
	> CREATE TABLE words_count_table (
		date VARCHAR(30) NOT NULL, 
		words_count JSON NOT NULL,
	    code VARCHAR(10) NOT NULL
	);
	INSERT INTO words_count_table (date, words_count, code) VALUES ('2023-06-01','{"col1": 3, "col2": 5, "col3": 20}', '000001');
	INSERT INTO words_count_table (date, words_count, code) VALUES ('2023-06-02', '{"col1": 5, "col2": 7}', '000001');
	INSERT INTO words_count_table (date, words_count, code) VALUES ('2023-06-04','{"col2": 5, "col3": 10}', '000001');
	INSERT INTO words_count_table (date, words_count, code) VALUES ('2023-06-05','{"col4": 13}', '000001');
	INSERT INTO words_count_table (date, words_count, code) VALUES ('2023-06-01','{"col1": 15, "col3": 25}', '000002');
	INSERT INTO words_count_table (date, words_count, code) VALUES ('2023-06-03','{"col6": 7, "col1": 5}', '000002');
	INSERT INTO words_count_table (date, words_count, code) VALUES ('2023-06-04','{"col7": 5}', '000002');
	
	
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
		open NUMERIC(7,2) NOT NULL,
		high NUMERIC(7,2) NOT NULL,
		low NUMERIC(7,2) NOT NULL,
		close NUMERIC(7,2) NOT NULL,
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
		df = df.fillna(0)
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

## STEP 3. Frontend

### Step 3-1. Code

* `components/GraphMulti.jsx`

	```jsx
	import React, { PureComponent } from 'react';
	import { ResponsiveContainer, ComposedChart, Line, Area, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

	function GraphMulti({ items, comWords, stockCode }) {
	const colors = ["#FF0060", "#0079FF", "#00DFA2", "#FFE79B", "#FF8400","#DD58D6", "#080202", "#40128B"]

	// Modify for multi graph
	const BarGraphs = comWords.map((comWord, i) => (
		<Bar yAxisId="left" type="monotone" dataKey={comWord[0]} fill={colors[i]} barSize={20} />
	));

	return (
	<div style={{ width: '70%', height: 300 }}>
		<ResponsiveContainer>
			<ComposedChart
				width={500}
				height={400}
				data={items}
				margin={{
				top: 20,
				right: 20,
				bottom: 20,
				left: 20,
				}}
			>
				<CartesianGrid stroke="#f5f5f5" />
				<XAxis dataKey="date" scale="band" />
				<YAxis yAxisId="left" />
				<YAxis yAxisId="stock" orientation="right" />
				<YAxis yAxisId="KOSPI" orientation="right" />
				<Tooltip />
				<Legend />
				<Area yAxisId="KOSPI" type="monotone" dataKey="KOSPI" fill="#408E91" stroke="#408E91" fillOpacity={0.03} />
				<Area yAxisId="stock" type="monotone" dataKey={stockCode} fill="#C04A82" stroke="#C04A82" fillOpacity={0.03} />
				{BarGraphs}// Modify for multi graph
			</ComposedChart>
		</ResponsiveContainer>
		</div>
	);
	}

	export default GraphMulti;
	```

* `pages/GraphPage/GraphPage.jsx`

	```jsx
	import GraphSearchForm from '../../components/GraphSearchForm';
	import GraphMulti from '../../components/GraphMulti';
	import GraphPie from '../../components/GraphPie';

	function GraphPage() {
		const [items, setItems] = useState([]);
		const [comWords,setComWords] = useState([]);
		const [code, setCode] = useState('');

	const handleSubmitSuccess = (res, stockCode) => {
	setItems(res.data);
	setComWords(res.comWords);
	setCode(stockCode)
		};

	return (
		<div>
			<h1>GraphPage</h1>
			<GraphSearchForm onSubmitSuccess={handleSubmitSuccess} />
			<GraphMulti items={items} comWords={comWords} stockCode={code}/>
			<GraphPie comWords={comWords} />
		</div>
		)
	}

	export default GraphPage;
	```

* `components/GraphSearchForm.jsx`

	```jsx
	...
	function GraphSearchForm({ onSubmitSuccess }) {
		...
		const submitHandler = async (e) => {
			e.preventDefault();
			const result = await loadItems({ stockCode, startDate, stopDate, reqCount });
			if (!result) return;
			onSubmitSuccess(result, stockCode);	// ADD
		};

		return (
			...
		);
	}
	```