---
layout: default
title: Fastapi
parent: Docker Format
grand_parent: Docker
nav_order: 3
---

# Fastapi(with docker)
{: .no_toc}

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

## STEP 1. Docker Code

### Step 1-1. File Structure

* **File structure**

  ```bash
  .
  ├── 📁backend
  │   ├── 📄Dockerfile
  │   ├── 📄main.py
  │   └── 📄requirements.txt
  └── 📄docker-compose.yml
  ```

* EC2에서 작업시 [EC2 Docker Engine Install] 먼저 수행

### Step 1-2. Docker Code

* `./docker-compose.yml`

  ```dockerfile
  version: '3.8'

  services:
    backend:
      container_name: backend
      build:
        context: ./backend/
        dockerfile: Dockerfile
      volumes:
        - ${PWD}/backend/:/usr/src/app
      ports:
        - 8000:8000
      command: ["uvicorn", "main:app","--host", "0.0.0.0", "--reload"]
  ```


* `./backend/Dockerfile`

  ```dockerfile
  FROM python:3.10

  WORKDIR /usr/src/app

  COPY ./requirements.txt /requirements.txt

  RUN pip install --upgrade pip
  RUN pip install --no-cache-dir --upgrade -r /requirements.txt
  ```

* `./backend/requirements.txt`

  ```
  fastapi
  uvicorn
  ```

* `./backend/main.py`

  ```python
  from fastapi import FastAPI

  app = FastAPI()

  @app.get("/")
  def hello():
      return {"message": "Hello World!"}
  ```

<br>

## STEP 2. fastapi 실행

### Step 2-1. Local에서 구동 시

* `bash`
  
  ```bash
  $ docker compose up -d --build  # build & run
  ```

* 인터넷 브라우저 `localhost:8000` or `localhost:8000/docs` 접속


### Step 2-2. AWS EC2에서 구동 시

* [해당 EC2] - [네트워킹] - [IPv4 방화벽] - [규칙추가]
  * Port or range : 8000 추가

* `bash`
  
  ```bash
  $ docker compose up -d --build  # build & run
  ```

* 인터넷 브라우저 `[EC2 Public IP]:8000` or `[EC2 Public IP]:8000/docs` 접속