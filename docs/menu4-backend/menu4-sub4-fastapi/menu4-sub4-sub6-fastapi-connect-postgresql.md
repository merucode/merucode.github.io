---
layout: default
title: FastApi Basic2
parent: FastApi
grand_parent: Backend
nav_order: 3
---

# FastApi Basic
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

## STEP 0. Site

* [fastapi 공식사이트](https://fastapi.tiangolo.com/ko/tutorial/)
* [https://blog.hanchon.live/guides/google-login-with-fastapi-and-jwt/](https://blog.hanchon.live/guides/google-login-with-fastapi-and-jwt/)

## STEP 1. Basic

### Step 1-1.  

```python
from fastapi import FastAPI  
from typing import Union

app = FastAPI()  

fake_items_db  =  [{"item_name":  "Foo"},  {"item_name":  "Bar"},  {"item_name":  "Baz"}]

### 경로 매개변수
@app.get("/users/{user_id}") 
async def read_user(user_id:  str):  
	return {"user_id":  user_id}

### 파일 경로 매개변수
@app.get("/files/{file_path:path}") 
async def read_file(file_path:str):  
	return {"file_path": file_path}

### 쿼리 매개변수(http://127.0.0.1:8000/items/?skip=0&limit=10)
@app.get("/items/")  
async def read_item(skip: int = 0, limit: int = 10): 
	return fake_items_db[skip  :  skip  +  limit]

### 선택적 매개변수(`item_id`가 경로 매개변수이고 `q`는 쿼리 매개변수 알아서 구별)
@app.get("/items/{item_id}")  
async def read_item(item_id: str, q: Union[str, None] = None): 
	if q: 
		return {"item_id":  item_id,  "q":  q}  
	return {"item_id":  item_id}

### 필수 쿼리 매개변수(http://127.0.0.1:8000/items/foo-item?needy=sooooneedy)
@app.get("/items/{item_id}") 
async def read_user_item(item_id: str, needy: str, skip: int = 0, limit: Union[int, None] = None ):  
	item = {"item_id": item_id, "needy": needy, "skip": skip, "limit": limit}  
	return  item
	# 3가지 쿼리 매개변수
	#`needy`, 필수적인 str
	#`str`.`skip`, 기본값이  `0`인  `int`
	#`limit`, 선택적인  `int`
```


### Step 1-2. Get Request Body

```python
from fastapi import FastAPI  
from pydantic import BaseModel 

class Item(BaseModel):  
	name: str  
	description: str | None  =  None  
	price: float  
	tax: float | None = None  
	
app = FastAPI()  

### POST로 JSON 데이터를 받으면 Item schemas에 의해 유효성검사 및 형태 변환 후
### item parameter로 저장
@app.post("/items/")  
async def create_item(item: Item):  
	item_dict = item.dict()  
	if item.tax:  
		price_with_tax = item.price + item.tax item_dict.update({"price_with_tax":  price_with_tax})  
		return  item_dict

### 경로, 쿼리 매개변수와 함께 사용하기
@app.put("/items/{item_id}")  
async def create_item(item_id: int, item: Item, q: str | None = None): 
	result = {"item_id": item_id, **item.dict()}  
	if q:
		result.update({"q":  q})
	return result

```

### Step 1-3. 쿼리 매개변수와 문자열 유효성 검사

```python
from typing import Annotated  
from fastapi import FastAPI, Query  

app = FastAPI()

@app.get("/items/")  
# q는 선택적이며 str이며, 최소 3자 ~ 최대 50자 유효성 필요
async def read_items(q: Annotated[str | None, Query(min_length=3, max_length=50)] = None):
	results  =  {"items":  [{"item_id":  "Foo"},  {"item_id":  "Bar"}]}  
	if q: 
		results.update({"q": q})  
	return  results
```

### Step 1-4. 경로 매개변수와 숫자 검증




