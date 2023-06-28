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
	    return await {"message":"Hello"}

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
