---
layout: default
title: Jupyter Notebook
parent: Docker
nav_order: 8
---

# Jupyter Notebook(with docker)
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

* **`Dockerfile`**

  ```dockerfile
  # docker hub에서 원하는 jupyter notebook 이미지 선택
  FROM jupyter/minimal-notebook:latest	
  
  WORKDIR /usr/src/app
  
  # docker jupyter notebook 권한 관련 환경변수 설정
  ENV CHOWN_EXTRA="/usr/src/app"
  ENV CHOWN_EXTRA_OPTS="-R"
  ```
  
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

* `Dockerfile`

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

* `docker-compose.yml`

  ```dockerfile
  version: '3.7'
  
  services:
    dev:
      build:
        context: ./
        dockerfile: Dockerfile
      volumes:
        - ${PWD}:/usr/src/app
      env_file:
        - ./.env
      ports:
        - 8888:8888
      user: root
  ```

  
