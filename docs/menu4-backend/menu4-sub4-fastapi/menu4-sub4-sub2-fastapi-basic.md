---
layout: default
title: FastApi Basic
parent: DRF
grand_parent: Backend
nav_order: 9
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

## STEP 0. Related Site

* [점푸 투 FastApi](https://wikidocs.net/book/8531)

<!------------------------------------ STEP ------------------------------------>

<br>

## STEP 1. Install and Create Project

### Step 1-1. Install Libraries

* `bash`

  ```bash
  $ python -m pip install --upgrade pip
  $ pip install fastapi
  $ pip install sqlalchemy
  $ pip install "uvicorn[standard]"
  ```
  * sqlalchemy : ORM
  * 유비콘(uvicorn) : 비동기 호출을 지원하는 파이썬용 웹 서버

### Step 1-2. Create Project

* `backend/myapi/main.py`
	```python
	from fastapi import FastAPI 
	
	app = FastAPI() 
	
	@app.get("/hello")  
	def hello(): 
		return {"message": "안녕하세요 파이보"}
	```
	
* `bash`
	```bash
	backend/myapi/$ uvicorn main:app --reload
	# main:app → main.py의 app객체 의미
	# --reload → 프로그램 변경 시 서버 재시작 없이 그 내용을 반영
	```
	* 8000번 포트로 FastAPI서버가 구동

* Connect to `http://127.0.0.1:8000/docs`
	* 실행 테스트: `/docs`
	* 읽기 문서: `/redoc`


### Step 1-3. Connect to Frontend

* `frontend/~`
	```react
	<script> 
	let message;
	fetch("http://127.0.0.1:8000/hello").then((response) => { 
		response.json().then((json) => {
		 message = json.message; 
		 }); 
	 }); 
	 </script>
	  
	 <h1>{message}</h1>
	```

* `backend/myapi/main.py`
	* CORS 예외 URL 등록
	```python
	from fastapi import FastAPI 
	from starlette.middleware.cors import CORSMiddleware # ADD
	
	app = FastAPI() 
	
	## ADD
	origins = [ 
	"http://localhost:3000"
	] 
	
	## ADD
	app.add_middleware( 	
		CORSMiddleware, 
		allow_origins=origins,
		allow_credentials=True, 
		allow_methods=["*"], 
		allow_headers=["*"], 
	)  
	
	@app.get("/hello")  
	def hello(): 
		return {"message": "안녕하세요 파이보"}
	```
 * Connect to `http://127.0.0.1:3000`

<!------------------------------------ STEP ------------------------------------>

<br>

## STEP 2. 
