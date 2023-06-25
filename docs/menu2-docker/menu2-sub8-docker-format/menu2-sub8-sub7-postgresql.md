---
layout: default
title: Jupyter Notebook
parent: Docker Format
grand_parent: Docker
nav_order: 2
---

# PostgreSQL(with docker)
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

## STEP 1. Dockerfile 생성

* `docker-compose.yml`

  ```dockerfile
  version: '3.8'
  
  services:
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
        - .env
  ```

* `database/Dockerfile`

  ```dockerfile
  FROM postgres:14.8
  ```

* `.env`

  ```
  INSTANCE_HOST= # EC2 static ip or database endpoint
  DB_USER=test_db
  POSTGRES_PASSWORD=test_password
  DB_NAME=test_user
  DB_PORT=5432
  ```

  * POSTGRES_PASSWORD 환경 변수 설정해줘야 접속 가능





----------------------------------------------------



<br>

## STEP 2. Dockerfile build 및 run

* **teminal**

  ```bash
  $ docker build -t jupyter .
  
  $ docker run \
      -v $PWD:/usr/src/app \
      -p 8888:8888 \
      --user root \
      jupyter
      
  # docker run 실행 시 아래와 같이 주소형식으로 token 값 나옴(...?token=token값)
  # Or copy and paste one of these URLs:
  # http://1039d10a8a77:8888/lab?token=8cf6f4302eff032c359c59fa95d71eca8fc108aeb1fbbb77
  ```

* 권한 관련 문제 발생 시 참고 사이트 : [jupyter-docker doc](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/troubleshooting.html)

 <br>

## STEP 3. Jupyter notebook 접속

* **인터넷 브라우저 `localhost:8888` 접속**
* **token 입력**

<br>

## STEP 4. Example Dockerfile Code

* **File structure**

  ```bash
  .
  ├── 📄docker-compose.yml
  ├── 📄.env
  └── 📁jupyter
      ├── 📄Dockerfile
      └── 📄requirements.txt
  ```
  
* `./jupyter/Dockerfile`

  ```dockerfile
  FROM jupyter/minimal-notebook:latest
    
  WORKDIR /usr/src/app
        
  # docker jupyter notebook 권한 관련 환경변수 설정
  ENV CHOWN_EXTRA="/usr/src/app"
  ENV CHOWN_EXTRA_OPTS="-R"
    
  COPY ./requirements.txt .
    
  # install pakages
  RUN pip install --upgrade pip
  COPY ./requirements.txt .
  RUN pip install -r requirements.txt
    
  # build : $ docker build -t jupyter .
  # run   : $ docker run -v $PWD:/usr/src/app -p 8888:8888 --user root jupyter
  ```

* `./docker-compose.yml`

  ```dockerfile
  version: '3.7'
    
  services:
    jupyter:
      container_name: jupyter
      build:
        context: ./jupyter/
        dockerfile: Dockerfile
      volumes:
        - ${PWD}/jupyter:/usr/src/app
      ports:
        - 8888:8888
      user: root
      # if you need to connect env_file(DB)
      env_file:
        - ./.env
  
  # build & run : docker compose up -d --build
  # token check : docker logs jupyter
  ```

  
