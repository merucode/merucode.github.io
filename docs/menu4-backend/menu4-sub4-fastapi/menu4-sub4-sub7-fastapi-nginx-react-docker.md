---
layout: default
title: FastApi Nginx React Docker
parent: FastApi
grand_parent: Backend
nav_order: 7
---

# FastApi Nginx React Docker
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

* [FastAPI 환경에서 Nginx 와 Uvicorn 을 통한 Deploy](https://sengwoolee.dev/133)
* [docker-compose + nginx.conf + React + FastAPI 로 HTTPS 통신, webSocket(SSL 적용) 구현하기](https://velog.io/@3436rngus/docker-compose-nginx.conf-React-FastAPI-%EB%A1%9C-HTTPS-%ED%86%B5%EC%8B%A0-webSocketSSL-%EC%A0%81%EC%9A%A9-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0)
* [Docker-Compose + Nginx + FastAPI](https://earthlyz9-dev.oopy.io/docker/docker-compose-nginx)


<br>

## STEP 1. Docker install on EC2

* Reference
  * [Docker]-[Docker-compose]-[Doceker-compose Install on EC2]

<br>

## STEP 2. fastapi-nginx

* [github : fastapi-nginx](https://github.com/merucode/form/tree/master/fastapi-nginx)

<br>

## STEP 3. fastapi-react-nginx

* From STEP 2.

* `./Dockerfile`

  ```docker
  FROM node:18-alpine3.16
    
  WORKDIR /usr/src/app
    
  RUN npm install -g npm@latest
  RUN npm install -g create-react-app
  ```

* `./docker-compose.yml`

  ```docker
  version: '3.8'

  services:
    backend:
      container_name: backend
      build:
        context: ./backend/
        dockerfile: Dockerfile
      volumes:
        - ${PWD}/backend/:/usr/src/app
      expose:
        - 8000
      command: ["uvicorn", "main:app", "--root-path", "/api", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers","--reload"]
    
    frontend:
      container_name: frontend
      build:
        context: ./
        dockerfile: Dockerfile
      volumes:
        - ${PWD}/:/usr/src/app/
      expose:
        - 3000
      command: /bin/sh   # run -it mode
      stdin_open: true   # run -it mode
      tty: true		   # run -it mode

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

* `nginx/nginx.conf`

  ```nginx
  # 3000번 포트에서 frontend가 돌아가고 있다는 것을 명시
  upstream frontend {
      server frontend:3000;
  }

  # 5000번 포트에서 backend서버가 돌아가고 있다는 것을 명시
  upstream backend {
      server backend:8000;
  }

  server {
      # nginx 서버 80번으로 열기
      listen 80;

      # 로케이션에는 우선 순위가 있는데 / 되는것만 우선순위가 가장 낮다. 
      # 그래서 여기서는 /api 로 시작하는 것을 먼저 찾고
      # 그게 없다면 / 이렇게 시작되는 것이니 요청을 http://frontend로
      location / {
          proxy_pass http://frontend;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header Host $host;
          proxy_redirect off;
      }
      
      # /api로 들어오는 요청을 http://backend로
      # rewrite를 사용함으로써 fastapi root_path를 이용하여
      # fastapi에서는 /api 주소를 생략하고 코드 작성 가능 
      location /api {
          rewrite ^/api/(.*)$ /$1 break;
          proxy_pass http://backend/api;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header Host $host;
          proxy_redirect off;
      }
  }
  ```

* `backend/main.py`

  ```python
  from fastapi import FastAPI

  app = FastAPI(root_path="/api")

  @app.get("/api")
  async def root():
    return {"Hello": "World!!"}
  ```

* `bash`

  ```bash
  $ docker compose up -d --build
  $ docker exec -it frontend /bin/sh
  /usr/src/app # create-react-app frontend
  ```

* Move `./Dockerfile` to `frontend/Dockerfile`

* `./docker-compose.yml`

  ```docker
  ...
    frontend:
      container_name: frontend
      build:
        context: ./frontend/
        dockerfile: Dockerfile
      volumes:
        - ${PWD}/frontend/:/usr/src/app/
      expose:
        - 3000
      command: [npm, run, start]
  ...
  ```

* `bash`

  ```bash
  $ docker compose up -d --build
  ```

* Check
  * connect to `[ec2_ip]` 
  * connect to `[ec2_ip]/api`, `[ec2_ip]/api/docs`