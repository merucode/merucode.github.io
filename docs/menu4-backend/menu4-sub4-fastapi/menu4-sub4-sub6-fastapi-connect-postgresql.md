---
layout: default
title: Fastapi와 기존 Postgresql Data 연결
parent: FastApi
grand_parent: Backend
nav_order: 3
---

# Fastapi와 기존 Postgresql Data 연결
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

* [https://jongsky.tistory.com/17](https://jongsky.tistory.com/17)

<br>

## STEP 1. 비동기 연결

### Step 1-1. Database 구조 확인

* 예제 데이터 : date, word, code로 구성

### Step 1-2. Code 구성

* `database.py`
	* 기존에 존재하는 database를 sqlalchemy를 이용해 연결해주는 파일
	```python
	import os
	from sqlalchemy import create_engine
	
	from sqlalchemy.ext.declarative import declarative_base
	from sqlalchemy.orm import sessionmaker
	
	 # Read ENV parameter
	db_host = os.environ["INSTANCE_HOST"] 
	db_user = os.environ["DB_USER"]
	db_pass = os.environ["DB_PASS"]
	db_name = os.environ["DB_NAME"]
	db_port = os.environ["DB_PORT"]

	SQLALCHEMY_DATABASE_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

	engine = create_engine(SQLALCHEMY_DATABASE_URL)
	SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
	
	Base = declarative_base()
	
	def get_db():
		db = SessionLocal()
		try:
			yield db
		finally:
		db.close()
	```

* `models.py` 
	* database.py에서 연결한 db를 테이블과 매핑시키는 역할
	```python
	from sqlalchemy import BIGINT, Column, Integer, String, Text
	from database import Base

	class WordsCount(Base):
		__tablename__ = "test_3"
		
		date = Column(Text, nullable=False, primary_key=True)
		word_counts = Column(Text, nullable=False)
		code = Column(Text, nullable=False)
	```

* `domain/words_count/words_count_router.py`
	*get 요청을 CRUD 처리를 위한 라우팅
	```python
	from fastapi import APIRouter, Depends
	from database import get_db
	from sqlalchemy.orm import Session
	
	from domain.words_count import words_count_crud

	router = APIRouter(
	    prefix="/words-count",
	)

	@router.get("/")
	def words_count(db:Session=Depends(get_db)):
		words_count = words_count_crud.get_words_count(db)
	    return words_count
	```

* `domain/words_count/words_count_crud.py`
	* 실제 CRUD 쿼리 처리
	```python	
	from sqlalchemy.orm import Session
	from models import WordsCount
	def get_words_count(db: Session):
		words_count = db.query(WordsCount).all()
	    return words_count
	```

* `main.py`
	* `words_count_router` 라우터 연결
	```python
	import os
	from fastapi import FastAPI, Request
	from starlette.middleware.cors import CORSMiddleware
	
	from domain.words_count import words_count_router
	
	frontend_url=os.environ["FRONTEND_URL"]
	origins = [ frontend_url ]
	
	app = FastAPI(root_path="/api")

	app.add_middleware(
	    CORSMiddleware,
	    allow_origins=origins,
	    allow_credentials=True,
	    allow_methods=["*"],
	    allow_headers=["*"],
	)
	
	app.include_router(words_count_router.router)
	```


## Step 3. Postgresql 비동기 연결(아직 미수행)

* start from `form-fastapi-react-basic`
* PostgreSQL 데이터베이스 비동기 처리 라이브러리
	* pip install asyncpg
*  
https://testdriven.io/blog/fastapi-sqlmodel/


