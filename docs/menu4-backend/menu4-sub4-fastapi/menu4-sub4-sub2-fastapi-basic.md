---
layout: default
title: FastApi Basic
parent: FastApi
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
  $ pip install alembic
  ```
  * sqlalchemy : ORM
  * 유비콘(uvicorn) : 비동기 호출을 지원하는 파이썬용 웹 서버
  * alembic : SQLAlchemy 모델 기반 데이터베이스 관리 도구

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
	backend/myapi/$ uvicorn main:app --host 0.0.0.0 --reload
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

* (**CORS 예외 URL 등록**)`backend/myapi/main.py`
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
			
### Step 2-3. Model로 Database 관리하기: Model 만들기

* `backend/models.py`
	```python
	from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
	from sqlalchemy.orm import relationship 
	
	from database import Base 
	
	class Question(Base): 
		__tablename__ = "question"
		  
		id = Column(Integer, primary_key=True)
		subject = Column(String, nullable=False) 
		content = Column(Text, nullable=False)
		create_date = Column(DateTime, nullable=False)
	
	class Answer(Base):
    	__tablename__ = "answer"
	
    	id = Column(Integer, primary_key=True)
    	content = Column(Text, nullable=False)
    	create_date = Column(DateTime, nullable=False)
    	question_id = Column(Integer, ForeignKey("question.id"))
    	question = relationship("Question", backref="answers")
	```
	* `primary_key` : Integer이고 기본키로 설정한 속성은 값이 자동으로 증가하는 특징(별도 설정 불필요)
	* `nullable=True` : Default
	* `relationship`: 관계
		* `backref`: 역관계

* `bash`
	```bash
	/backend/myapi/$ alembic init migrations
	```
	* `migrations` directroy : alembic 리비전 파일 저장
	* `alembic.ini` : alembic 환경 설정 파일

* `backend/myapi/alembic.ini`
	```
	...
	sqlalchemy.url = sqlite:///./myapi.db 
	...
	```
	* alembic이 사용할 데이터베이스의 접속주소를 설정

* `backend/myapi/migrations/env.py`
	```python
	... 
	import models 
	...
	# add your model's MetaData object here  
	# for 'autogenerate' support  
	# from myapp import mymodel  
	# target_metadata = mymodel.Base.metadata 
	target_metadata = models.Base.metadata # Add
	...
	```

* `bash`
	```bash
	myapi/$ alembic revision --autogenerate
	# 리비전 파일 생성
	myapi/$ alembic upgrade head
	# 리비전 파일 실행
	```

<!------------------------------------ STEP ------------------------------------>

<br>

## STEP 3. Api 구성하기

* 질문 목록 api 만들기

### Step 3-1. Router

* `myapi/domain/question/question_router.py`
	```python
	from fastapi import APIRouter 
	
	from database import SessionLocal 
	from models import Question 
	
	router = APIRouter( 
		prefix="/api/question", 
	) 
	
	@router.get("/list")  
	def question_list(): 
		db = SessionLocal() 
		_question_list = db.query(Question).order_by(Question.create_date.desc()).all() 
		db.close() # 세션 종료가 아닌 세션을 커넥션 풀에 반환
		return _question_list
	```
	* 라우팅 : FastAPI가 요청받은 URL을 해석하여 그에 맞는 함수를 실행하여 그 결과를 리턴하는 행위
	* router 객체를 생성하여 FastAPI 앱에 등록
	* `/api/question/list` 요청 → `/api/question` prefix router → `/list` 등록된 `question_list` 실행

* `myapi/main.py`
	```python
	from fastapi import FastAPI 
	from starlette.middleware.cors import CORSMiddleware 
	
	from domain.question import question_router  # Add
	app = FastAPI() 
	origins = [ ... ] 
	app.add_middleware( ... ) 
	
	app.include_router(question_router.router) # Add
	```
* Connect to `http://127.0.0.1:8000/docs`


### Step 3-2. Dependency Injection
* db 세션 객체를 생성하고 종료하는 이런 반복적인 작업을 깔끔하게 처리하는 방법
* `myapi/database.py`
	```python
	...
	def get_db(): 
		db = SessionLocal() 
		try: 
			yield db 
		finally: 
			db.close()
	```
* `myapi/domain/question/question_router.py`
	```python
	from fastapi import APIRouter, Depends 	# Add
	from sqlalchemy.orm import Session		# Add
	
	from database import get_db
	...
	@router.get("/list")  
	def question_list(db: Session = Depends(get_db)):
			_question_list = db.query(Question).order_by(Question.create_date.desc()).all()
		return _question_list
	```
	* `db: Session` : db 객체가 Session 타입 의미


### Step 3-3. 스키마
* **Pydantic**
	* FastAPI의 입출력 스펙을 정의하고 그 값을 검증
	* [https://pydantic-docs.helpmanual.io/](https://pydantic-docs.helpmanual.io/)
		*  입출력 항목의 갯수와 타입을 설정
		*  입출력 항목의 필수값 체크
		*   입출력 항목의 데이터 검증
* **스키마** : 보통 데이터의 구조와 명세
* (**Pydantic 스키마 작성하기**)`myapi/domain/question/question_schema.py`
	
	```python
	import datetime 
	from pydantic import BaseModel 
	
	class Question(BaseModel): 
		id: int 
		subject: str 
		content: str 
		create_date: datetime.datetime
		
		# Question 모델 항목이 자동으로 스키마로 매핑
		class Config: 
			orm_mode = True
	```
	* 4개의 필수항목으로 구성
	* 만약 필수항목이 아니게 설정하려면
		* `subject: str | None = None`
	
* (**라우터에 Pydantic 적용**)`myapi/domain/question/question_router.py`
	```python
	...
	from domain.question import question_schema
	...
	@router.get("/list", response_model=list[question_schema.Question]) # Update 
	def question_list(db: Session = Depends(get_db)): 
		_question_list = db.query(Question).order_by(Question.create_date.desc()).all() 
		return _question_list
	```
	* `response_model=list[question_schema.Question]`
		* question_list 함수의 리턴값은 Question 스키마로 구성된 리스트임을 의미
	* Question 스키마에서 content 항목을 제거한다면 질문 목록 API의 출력 항목에도 content 항목이 제거
		* 리턴되는 `_question_list`를 수정할 필요가 없음


### Step 3-4. CRUD
* `myapi/domain/question/question_crud.py`
	```python
	from models import Question 
	from sqlalchemy.orm import Session 
	
	def get_question_list(db: Session): 
		question_list = db.query(Question)\
			.order_by(Question.create_date.desc())\
			.all() 
		return question_list
	```
	
* `myapi/domain/question/question_router.py`
	```python
	from fastapi import APIRouter, Depends 
	from sqlalchemy.orm import Session 
	
	from database import get_db 
	from domain.question import question_schema, question_crud # Update
	
	router = APIRouter(
		prefix="/api/question", 
	) 
	
	@router.get("/list", response_model=list[question_schema.Question])
	def question_list(db: Session = Depends(get_db)): 
		_question_list = question_crud.get_question_list(db)  # Update
		return _question_list
	```
	<br>

## STEP 4. 비동기 방식으로 구성하기

### Step 4-1. FastApi 와 async
* 동기 Example
	```python
	results = await some_library()
	```
	
* 비동기 Example
	```python
	@app.get('/')  
	async def read_results(): 
		results = await some_library() 
		return results
	```

### Step 4-2. 비동기로 구현하기

* `bash`
	```bash
	myapi/$ pip install aiosqlite
	```
	* sqlite3 : `pip install aiosqlite`
	* sqlalchemy 버전도 1.4+ 이상의 버전을 사용

* `myapi/database.py` 
	```python
	...
	from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
	...
	async_engine = create_async_engine("postgresql+asyncpg://user:password@postgresserver/db")
	
	async def get_async_db(): 
		db = AsyncSession(bind=async_engine) 
		try: 
			yield db 
		finally: 
			await db.close()
	```
	* `+asyncpg` 를 붙인 데이터베이스 주소를 사용
	*  `db.query(Query)` 대신 `db.execute(select(Query))`와 같은 방식을 사용
	
* `myapi/domain/question/question_crud.py`
	```python
	...
	from sqlalchemy import select
	...
	async def get_async_question_list(db: Session): 
		data = await db.execute(select(Question) 
			.order_by(Question.create_date.desc()) 
			.limit(10)) 
		return data.all()
	```

