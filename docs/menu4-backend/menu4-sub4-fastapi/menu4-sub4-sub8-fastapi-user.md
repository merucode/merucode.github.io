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
