---
layout: default
title: React
parent: Docker Format
grand_parent: Docker
nav_order: 3
---

# React(with docker)
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

* [프론트엔드 개발자를 위한 Docker로 React 개발 및 배포하기](https://velog.io/@oneook/Docker%EB%A1%9C-React-%EA%B0%9C%EB%B0%9C-%EB%B0%8F-%EB%B0%B0%ED%8F%AC%ED%95%98%EA%B8%B0)
* [Github](https://github.com/merucode/form/tree/react_basic)


## STEP 1. Docker Code

### Step 1-1. Create React App

* EC2에서 작업시 [EC2 Docker Engine Install] 먼저 수행

* **File structure**

  ```bash
  .
  ├── 📄Dockerfile
  ├── 📄docker-compose.yml
  └── 📁frontend(create)
  ```
  
- **`Dockerfile`**

  ```dockerfile
  FROM node:18-alpine3.16
  
  WORKDIR /usr/src/app
  
  RUN npm install -g npm@latest
  RUN npm install -g create-react-app
  ```

- **`docker-compose.yml`**

  ```dockerfile
  version: '3.8'
    
  services:
    frontend:
      container_name: frontend
      build:
        context: ./
        dockerfile: Dockerfile
      volumes:
        - ./:/usr/src/app/
      ports:
        - 3000:3000
      command: /bin/sh # run -it mode
      stdin_open: true # run -it mode  
      tty: true        # run -it mode
  ```

- **`bash`(create react  app and dev server run)**

  ```bash
  $ docker compose up -d --build
  $ docker exec -it frontend /bin/sh
  /usr/src/app # create-react-app frontend
  
  /usr/src/app # cd frontend
  /usr/src/app/frontend # npm run start
  ```

  - you can connent to `http://localhost:3000`
  - EC2 구동시
    - [해당 EC2] - [네트워킹] - [IPv4 방화벽 규칙] - [규칙 추가]
      - port 3000 추가
    - connect to `[EC2 public ip]:3000`


### Step 1-2. Docker Code

* **File structure**

  ```bash
  .
  ├── 📄docker-compose.yml
  └── 📁frontend
      ├── 📄Dockerfile
      ├── 📁public
      └── 📁src
  ```

* `bash`

  ```bash
  $ mv Dockerfile frontend/
  ```

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
      ports:
        - 3000:3000
      command: sh -c "npm install && npm run start"
  ```

  * npm install : After you operate git pull, install node_modules 

* `frontend/Dockerfile`

  ```
  FROM node:18-alpine3.16
  
  WORKDIR /usr/src/app
  ```

<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 2. Install Dependencies

- **`bash`**

  ```bash
  $ docker compose up -d --build
  $ docker exec -it frontend /bin/sh
  /usr/src/app # cd frontend
  /usr/src/app/frontend # npm install react-router-dom@6
  /usr/src/app/frontend # npm install styled-components@5
  ```


<br>


<!------------------------------------ STEP ------------------------------------>
