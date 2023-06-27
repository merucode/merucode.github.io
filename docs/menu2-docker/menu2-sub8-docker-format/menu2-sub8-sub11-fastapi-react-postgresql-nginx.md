---
layout: default
title: fastapi-react-postgresql-nginx
parent: Docker Format
grand_parent: Docker
nav_order: 11
---

# fastapi-react-postgresql-nginx
{: .no_toc .d-inline-block }
ing
{: .label .label-green }

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

* [Github]
  * Setting ENV 
    * `/frontend/.env`
    * `.database.env`
    * `.backend.env`


<br>

## Step 1. Basic Library Install

### Step 1-1. File Structure

* **File structure**

  ```bash
  .
  ├── 📁backend
  │   ├── 📁domain 
  │   ├── 📄models.py
  │   ├── 📄database.py
  │   ├── 📄Dockerfile
  │   ├── 📄main.py
  │   └── 📄requirements.txt
  ├── 📁database
  │   ├── 📄Dockerfile
  │   └── 📁postgresql
  ├── 📄docker-compose.yml
  ├── 📄.backend.env
  ├── 📄.database.env
  ├── 📁frontend
  │   ├── 📄Dockerfile
  │   ├── 📁node_modules
  │   ├── 📄package-lock.json
  │   ├── 📄package.json
  │   ├── 📄.env
  │   ├── 📁public
  │   └── 📁src
  └── 📁nginx
      ├── 📄Dockerfile
      └── 📄nginx.conf
  ```

### Step 1-2. Install Basic Docker format
* [react](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub3-react.html)
* [fastapi](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub4-fastapi.html)
* [postgresql](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub7-postgresql.html)


<br>

### Step 1-2. Install Basic Docker format
* [react](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub3-react.html)
* [fastapi](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub4-fastapi.html)
* [postgresql](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub7-postgresql.html)

<br>

## STEP 2. Connect using Nginx

### Step 2-1. Nginx Docker Code

* `docker-compose.yml`
	```dockerfile
	...
	  nginx:
	    container_name: nginx
	    build:
	      context: ./nginx/
	      dockerfile: Dockerfile
	    ports:
	      - 80:80
	    depends_on:
	      - backend
	      - frontend
	```

* `nginx/Dockerfile`

	```dockerfile
	FROM nginx:1.25-alpine

	# Delete conf and copy
	RUN rm /etc/nginx/conf.d/default.conf
	COPY nginx.conf /etc/nginx/conf.d/
	```

* `nginx/nginx.conf`

	```nginx
	# frontend : 3000 port
	upstream frontend {
	    server frontend:3000;
	}
	
	# backend : 8000 port
	upstream backend {
	    server backend:8000;
	}
	
	server {
	    listen 80;
	
	    # When request in "/", connect to "http://frontend" 
	    location / {
	        proxy_pass http://frontend;
	        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
	        proxy_set_header Host $host;
	        proxy_redirect off;
	    }
	
	    # When request in "/api", connect to "http://backend" 
	    # Using rewrite and fastapi root_path, handling 'api/' as '/' in backend(fastapi)
	    location /api {
	    	rewrite ^/api/(.*)$ /$1 break;
	        proxy_pass http://backend/api;
	        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
	        proxy_set_header Host $host;
	        proxy_redirect off;
	    }
	}
	```

### Step 2-2. Connect Nginx

* `docker-compose.yml`

	```dockerfile
	version: '3.8'
	
	services:
	  frontend:
	    container_name: frontend
	    build:
	      context: ./frontend/
	      dockerfile: Dockerfile
	    volumes:
	      - ${PWD}/frontend/:/usr/src/app/
	    expose:
	      - 3000
	    command: sh -c "npm install && npm run start"
	
	  backend:
	    container_name: backend
	    build:
	      context: ./backend/
	      dockerfile: Dockerfile
	    volumes:
	      - ${PWD}/backend/:/usr/src/app
	    expose:
	      - 8000
	    command: ["uvicorn", "main:app","--root-path", "/api", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--reload"]
	
	  database:
	    container_name: database
	    build:
	      context: ./database/
	      dockerfile: Dockerfile
	    volumes:
	      - ${PWD}/database/postgresql:/var/lib/postgresql/data/
	    expose:
	      - 5432
	    env_file:
	      - .database.env
	
	  nginx:
	  ...
	```

	* ports → expose 변경(nginx에서 분배)
	* backend command 변경
		* `—proxy-header` : nginx 에서 프록시 패스로 해당 어플리케이션을 연결하고 싶을 때 추가해주는 옵션
		* `—root-path` : api uri의 prefix 를 설정함
 * connect to `http://[EC2 Public IP]`, `http://[EC2 Public IP]/api/`

<br>

## STEP 3. Connect Backend and Frontend


### Step 3-1. Frontend

* **File Structure**

	```bash
	📁frontend
	├── 📄Dockerfile
	├── 📁node_modules
	├── 📄package-lock.json
	├── 📄package.json
	├── 📄.env
	├── 📁public
	└── 📁src
		├── 📄Main.js
		├── 📄index.js
		├── 📄urls.js
		├── 📁components
		│   ├── 📄App.js
		│   └── 📄Header.jsx
		└── 📁pages
		    └── 📄HomePage
	```

* `bash`
	*	Install `axios`, `react-router-dom@6`
	```bash
	$ docker compose up -d --build
	$ docker exec -it frontend /bin/sh
	# npm install axios react-router-dom@6 --save
	```
* Make as [React-router-basic-form]()

* `.env`

	```python
	REACT_APP_BACKEND_URL=http://[EC2 Public IP]/api/
	# REACT_APP_BACKEND_URL=http://13.124.156.36/api/
	```

* `urls.js`

	```jsx
	export const BACKEND_URL = process.env.REACT_APP_ACKEND_URL
	```

* `pages/HomePage/HomePage.jsx`

	```jsx
	import axios from "axios";
	import { useEffect, useState } from 'react';
	
	import { BACKEND_URL } from '../../urls';
	
	function HomePage() {
		const [update,setUpdate] = useState('');
		let message;
	
		useEffect(() => {
	    axios.get(BACKEND_URL)
	      .then((res) => {
	        message = res.data.message
	        setUpdate(message)
	      })
		  }, [update]);
	
	    return (
	    <div>
		    <h1>HomePage</h1>
		    <div>Data From Backend : </div>
		    <div>{update}</div>
		</div>
	    );
	}
	
	export default HomePage;
	```




### Step 3-2. Backend

* N/A

<br>



## STEP 4. Connect with Database



### Step 4-1. Create Data Table

* `bash`

  ```bash
  $ docker compose up -d --build
  $ docker exec -it database /bin/bash
  > su - postgres
  > psql -U test_user -d test_db;
  > CREATE TABLE test_table (
  	id SERIAL PRIMARY KEY,
      name VARCHAR(20) NOT NULL,
  	date DATE NOT NULL, 
  	value INT NOT NULL
  );
  > INSERT INTO test_table (name, date, value) VALUES ('KIM', '2023-06-27', 10);
  > INSERT INTO test_table (name, date, value) VALUES ('LEE', '2023-06-20', 50);
  ```



### Step 4-2. Database

* `docker-compose.yml`

  ```dockerfile
  ...
    database:
      ...
      ports:			# replace expose
        - 5432:5432
  ...
  ```

* [EC2] - [네트워킹] - [IPv4 방화벽] - [규칙추가]

  * port 5432 추가 

    

### Step 4-3. Backend

* We use `async` connecting method

* `.backend.env`

  ```bash
  INSTANCE_HOST= # EC2 public IP
  DB_USER=test_user
  DB_PASS=test_password
  DB_NAME=test_db
  DB_PORT=5432
  
  SECRET_KEY=test_secret_key
  FRONTEND_URL=# EC2 public IP
  ```

* `docker-compose.yml`

  ```dockerfile
  ...
    backend:
      ...
      env_file:
  	  - .backend.env
  ...
  ```

* `requirements.txt`

  ```
  ...
  sqlalchemy  
  psycopg2
  asyncpg
  ```

* `models.py`

  ```python
  from sqlalchemy import Column, Integer, String, Text, Date
  from database import Base
  
  class TestModel(Base):
      __tablename__ = "test_table"
  
      id = Column(Integer, primary_key=True)
          date = Column(Date, nullable=False)
          name = Column(String, unique=True, nullable=False)
          value = Column(Integer, nullable=False)
  ```

* `database.py`

  ```python
  import os
  from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine # Async
  from sqlalchemy.orm import declarative_base            
  
  db_host = os.environ["INSTANCE_HOST"]  # Read ENV file in Docker compose
  db_user = os.environ["DB_USER"]  
  db_pass = os.environ["DB_PASS"]
  db_name = os.environ["DB_NAME"] 
  db_port = os.environ["DB_PORT"]
  
  SQLALCHEMY_DATABASE_URL = f"postgresql+asyncpg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}" # Async
  engine = create_async_engine(SQLALCHEMY_DATABASE_URL, echo=True) # Async
  Base = declarative_base()
  
  # Async
  async def get_async_db() -> AsyncSession:
      db= AsyncSession(bind=engine)
      try:
          yield db
      finally:
          await db.close()
  ```

* `domain/test/test_crud.py`

  ```python
  from sqlalchemy.ext.asyncio import AsyncSession # Async
  from sqlalchemy import select # Async
  
  from models import TestModel
  
  async def get_data(db: AsyncSession):
      data = await db.execute(select(TestModel))
      result = data.scalars().fetchall()   # Load data from PG
      return result
  ```

* `domain/test/test_router.py`

  ```python
  from typing import Union
  from fastapi import APIRouter, Depends
  
  from database import get_async_db # Async
  from sqlalchemy.ext.asyncio import AsyncSession # Async
  
  from domain.test import test_crud
  
  router = APIRouter(
      prefix="/test",
  )
  
  @router.get("/")
  async def test_get_data(db:AsyncSession=Depends(get_async_db)):
      result = await test_crud.get_data(db)
      return result
  ```

* `main.py`

  ```python
  ...
  from domain.test import test_router
  ...
  app.include_router(test_router.router)
  ```

* connect `[EC2 Public IP]/api/test/`
