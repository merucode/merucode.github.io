---
layout: default
title: Jupyter Notebook
parent: Dockerfile Format
grand_parent: Docker
nav_order: 2
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

## STEP 0. Reference Site

* 권한 관련 문제 발생 시 참고 사이트 : [jupyter-docker doc](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/troubleshooting.html)
* Github : [https://github.com/merucode/form/tree/jupyter_basic](https://github.com/merucode/form/tree/jupyter_basic)

<br>

## STEP 1. Dockerfile Code

* EC2에서 작업시 `[EC2 Docker Engine Install]` 먼저 수행

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
  FROM jupyter/base-notebook:latest
    
  WORKDIR /usr/src/app
        
  # Setting ENV for docker jupyter notebook
  ENV CHOWN_EXTRA="/usr/src/app"
  ENV CHOWN_EXTRA_OPTS="-R"
    
  # install pakages
  RUN pip install --upgrade pip
  COPY ./requirements.txt .
  RUN pip install -r requirements.txt
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
      env_file:
        - ./.env
  
  # build & run : docker compose up -d --build
  # token check : docker logs jupyter
  ```

<br>

## STEP 2. Jupyter notebook 접속

### Step 2-1. Local에서 구동 시

* `bash`
  
  ```bash
  $ docker compose up -d --build  # build & run
  $ docker logs jupyter           # token check
  ```

* 인터넷 브라우저 `localhost:8888` 접속
* token 입력


### Step 2-2. AWS EC2에서 구동 시

* [해당 EC2] - [네트워킹] - [IPv4 방화벽] - [규칙추가]
  * Port or range : 8888 추가

* `bash`
  
  ```bash
  $ docker compose up -d --build  # build & run
  $ docker logs jupyter           # token check
  ```

* 인터넷 브라우저 `[EC2 Public IP]:8888` 접속
* token 입력