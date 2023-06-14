---
layout: default
title: FastApi User
parent: FastApi
grand_parent: Backend
nav_order: 8
---

# FastApi User
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

* Start from `fastapi-nginx-docker`

## STEP 1. Make User Model

### Step 1-1. Make Structure
* `database.py`
* `models.py`
* `domain` directory
	* `user` directory
		* `user_router.py`


### Step 1-2. Setting

* `database.py`

	```python
	import os
	from sqlalchemy import create_engine
	from sqlalchemy.ext.declarative import declarative_base
	from sqlalchemy.orm import sessionmaker

	db_host = os.environ["INSTANCE_HOST"]  # Read ENV file in Docker compose
	db_user = os.environ["DB_USER"]  
	db_pass = os.environ["DB_PASS"]
	db_name = os.environ["DB_NAME"] 
	db_port = os.environ["DB_PORT"]

	# SQLALCHEMY_DATABASE_URL = "sqlite:///./myapi.db"
	# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"
	SQLALCHEMY_DATABASE_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

	engine = create_engine(SQLALCHEMY_DATABASE_URL)

	SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

	Base = declarative_base()
	```

* `models.py`

	```python
	from sqlalchemy import Column, Integer, String, Text

	from database import Base

	class User(Base):
	    __tablename__ = "user"

	    id = Column(Integer, primary_key=True)
	    email = Column(String, unique=True, nullable=False)
	    password = Column(String, nullable=False)
		username = Column(String, unique=True, nullable=False)
	```

* `.env`

	```
	INSTANCE_HOST=
	DB_USER=
	DB_PASS=
	DB_NAME=
	DB_PORT=
	```

* `docker-compose.yml`

	```docker
	...
	  backend:
	    ...
	    env_file:
	      - .env
	      - 
	...
	```

	
### Step 1-3. Initalize alemic

* `/backend/requirements.txt`

	```python
	fastapi
	uvicorn

	sqlalchemy	# Add
	alembic		# Add
	psycopg2	# Add
	```

* `bash`
	
	```bash
	$ docker compose up -d --build
	$ docker exec -it backend /bin/bash
	> alembic init migrations
	> exit
	$ docker compose down
	```

* `backend/alembic.ini`

	```python
	...
	# sqlalchemy.url = sqlite:///./myapi.db # 주석처리
	...
	```

* `backend/migraions/env.py`

	```python
  import os
	...
	import models	# Add
	...
	config = context.config
	
	# Read ENV file in Docker compose
	db_host = os.environ["INSTANCE_HOST"] 
	db_user = os.environ["DB_USER"]  
	db_pass = os.environ["DB_PASS"]
	db_name = os.environ["DB_NAME"] 
	db_port = os.environ["DB_PORT"]
	
	SQLALCHEMY_DATABASE_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
	config.set_main_option('sqlalchemy.url', SQLALCHEMY_DATABASE_URL)
	...
	target_metadata = models.Base.metadata # Add
	```

### Step 1-4. Model Migrate

* `bash`
```bash
$ docker compose up -d --build
$ docker exec -it backend /bin/bash
> cd migrations
> mkdir versions
# versions 폴더 미생성 상태에서 revision 수행 시 에러발생
> cd ..
> alembic revision --autogenerate -m "0001_make_user"
> alembic upgrade head
```

<br>

## STEP 2. Make Function to Create User

### Step 2-0. Connect React

* [fastapi nginx react docker]

### Step 2-1. Make Function on Backend

* `backend/domain/user/`
	* user_schema.py
	* user_route.py
	* user_crud.py

* `backend/requirements.txt`

	```
	...
	pydantic[email]		# chekcing email
	passlib[bcrypt]		# hashing passward
	```

* `backend/domain/user/user_route.py`

	```python
	from fastapi import APIRouter, HTTPException
	from fastapi import Depends
	from sqlalchemy.orm import Session
	from starlette import status

	from database import get_db
	from domain.user import user_crud, user_schema

	router = APIRouter(
		prefix="/user",
	)

	@router.post("/create", status_code=status.HTTP_204_NO_CONTENT)
	def user_create(_user_create: user_schema.UserCreate, db: Session = Depends(get_db)):	# schema(UserCreate) 형식 비교용으로 사용
		# 동일 user 확인 crud 실행(schema로도 비교 되나(uniqe) 에러 메시지를 위해 사용)
		user = user_crud.get_existing_user(db, user_create=_user_create)  
    	if user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="이미 존재하는 사용자입니다.")
		# user 생성 crud 실행
		user_crud.create_user(db=db, user_create=_user_create)
	```

* `backend/domain/user/user_crud.py`

	```python
	from passlib.context import CryptContext	# For hashing password
	from sqlalchemy.orm import Session
	from domain.user.user_schema import UserCreate
	from models import User						# 데이터 저장을 위한 Model

	pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")	# hashing password

	# user 생성
	def create_user(db: Session, user_create: UserCreate):	# schema(UserCreate) 형식 비교용으로 사용
		# post data를 mapping 한 모델 class 인스턴스 생성 
		# passward는 hashing value 사용
		db_user = User(username=user_create.username,
					password=pwd_context.hash(user_create.password1),
					email=user_create.email)
		db.add(db_user) # db에 db_user 추가
		db.commit()		# db 커밋

	# 동일 user 확인
	def get_existing_user(db: Session, user_create: UserCreate):
    return db.query(User).filter(
        (User.username == user_create.username) |
        (User.email == user_create.email)
    ).first()
	```

* `backend/domain/user/user_schema.py`

	```python
	from pydantic import BaseModel, validator, EmailStr

	class UserCreate(BaseModel):
		username: str
		password1: str
		password2: str
		email: EmailStr

		@validator('username', 'password1', 'password2', 'email')
		def not_empty(cls, v):
			if not v or not v.strip():
				raise ValueError('빈 값은 허용되지 않습니다.')
			return v

		@validator('password2')
		def passwords_match(cls, v, values):
			if 'password1' in values and v != values['password1']:
				raise ValueError('비밀번호가 일치하지 않습니다')
			return v
	```
	* passwords_match 메서드의 values 매개변수에는 UserCreate의 속성들이 `변수명:값, ...` 형태로 전달

* `backend/main.py`

	```python
	...
	from domain.user import user_router
	...
	app.include_router(user_router.router)
	```

* `backend/database.py`

	```python
	...
	def get_db():
		db = SessionLocal()
		try:
			yield db
		finally:
			db.close()
	```

* connect `ec2_ip/api/docs` and Check function



### Step 2-2. Make Function on Frontend



<br>



## Step 3. Login/Logout



### Step 3-1. Login on Backend

* `backend/requirement.txt`

  ```
  ...
  python-multipart
  python-jose[cryptography]
  ```

* `backend/domain/user/user_schema.py`

  ```python
  ...
  # We don't need to make input schema on login api, 
  # because OAuth2PasswordRequestForm class offer 
  # So, only make output schema on login api
  class Token(BaseModel):
      access_token: str
      token_type: str
      username: str
  ```

* `backend/domain/user/user_crud.py`

  ```python
  ...
  def get_user(db: Session, username: str):
      return db.query(User).filter(User.username == username).first()
  ```

* `backend/domain/user/user_router.py`

  ```python
  import os
  from datetime import timedelta, datetime
  ...
  from fastapi.security import OAuth2PasswordRequestForm
  from jose import jwt
  ...
  from domain.user.user_crud import pwd_context
  
  ### Access Token
  ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
  SECRET_KEY = os.environ["SECRET_KEY"]
  ALGORITHM = "HS256"
  
  ### Login
  @router.post("/login", response_model=user_schema.Token)
  def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
                             db: Session = Depends(get_db)):
  
      # check user and password
      user = user_crud.get_user(db, form_data.username) # get user data
      # verify user and passward
      if not user or not pwd_context.verify(form_data.password, user.password):
          raise HTTPException(
              status_code=status.HTTP_401_UNAUTHORIZED,
              detail="Incorrect username or password",
              headers={"WWW-Authenticate": "Bearer"},
          )
  
      # make access token
      data = {
          "sub": user.username,
          "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
      }
      access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
  
      return {
          "access_token": access_token,
          "token_type": "bearer",
          "username": user.username
      }
  ```

* `./.env`

  ```
  SECRET_KEY=[new_secret_key]
  ```

  







### Step 3-2. Login on Frontend



### Step 3-3. Logout on Frotend

