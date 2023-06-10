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

## STEP 2. Basic Structure

### Step 2-1. FastAPI 프로젝트 구조

```
├── frontend
└── backend
	├── main.py 
	├── database.py
	├── models.py
	└── domain 
	   ├── answer
	   ├── question
	   └── user
```

* **`main.py`**
	* 파일에 생성한 app 객체는 FastAPI의 핵심 객체
	* FastAPI 프로젝트의 전체적인 환경을 설정하는 파일
* **`database.py`**
	* 데이터베이스를 사용하기 위한 변수, 함수등을 정의하고 접속할 데이터베이스의 주소와 사용자, 비밀번호등을 관리
* **`models.py`**
	* SQLAlchemy는 모델 기반으로 데이터베이스를 처리
* **Api를 구성하는 domain 디렉터리**
	* question, anser, user 3개의 도메인 구성
	* 각 도메인은 api를 생성하기 위해 다음의 파일을 필요
		* 라우터 파일(`question_router.py`)
		* 데이터베이스 처리 파일(`question_crud.py`)
		* 입출력 관련 파일(`question_schema.py`)

### Step 2-2. Model로 Database 관리하기: Database 설정

* `backend/database.py`
	* sqlalchemy 등록
		```python
		from sqlalchemy import create_engine 
		from sqlalchemy.ext.declarative import declarative_base 
		from sqlalchemy.orm import sessionmaker
		
		# 데이터베이스 접속 주소
		SQLALCHEMY_DATABASE_URL = "sqlite:///./myapi.db" 
		
		
		engine = create_engine(
			SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} 
		) 
		
		# 데이터베이스에 접속하기 위해 필요한 클래스
		SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) 
		# 데이터베이스 모델을 구성할 때 사용되는 클래스
		Base = declarative_base()
		```
		* `autocommit`
			* `False` :`commit` 해야 database에 변경사항 적용(rollback으로 되돌릴 수 있음)
			* `True`: 바로 database에 변경사항 반영(rollback 불가)
			
### Step 2-3.  Model로 Database 관리하기: Model 만들기
* `backend/models.py`
	```python
	from sqlalchemy import Column, Integer, String, Text, DateTime 
	from database import Base 
	
	class Question(Base): 
		__tablename__ = "question"
		  
		id = Column(Integer, primary_key=True)
		subject = Column(String, nullable=False) 
		content = Column(Text, nullable=False)
		create_date = Column(DateTime, nullable=False)
	```

...