---
layout: default
title: React Native
parent: Docker Format
grand_parent: Docker
nav_order: 4
---

# React Native(with docker)
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

* [Docs](https://reactnative.dev/docs/environment-setup)



<br>




## STEP 1. Docker Code

### Step 1-1. Create React App

* EC2에서 작업시 [EC2 Docker Engine Install] 먼저 수행

* **File structure**

  ```bash
  .
  ├── 📄Dockerfile
  ├── 📄docker-compose.yml
  └── 📁myapp(create)
  ```
  
- **`Dockerfile`**

  ```dockerfile
  FROM node:18-alpine3.16
  
  WORKDIR /usr/src/app
  
  RUN npm install -g npm@latest
  ```

- **`docker-compose.yml`**

  ```dockerfile
  version: '3.8'
    
  services:
    app:
      container_name: app
      build:
        context: ./
        dockerfile: Dockerfile
      volumes:
        - ./:/usr/src/app/
      ports:
        - 19000:19000
        - 19001:19001
        - 19002:19002
      command: /bin/sh # run -it mode
      stdin_open: true # run -it mode  
      tty: true        # run -it mode
  ```

- **`bash`(create react  app and dev server run)**

  ```bash
  $ docker compose up -d --build
  $ docker exec -it frontend /bin/sh
  /usr/src/app # npx create-expo-app myapp
  
  /usr/src/app # cd myapp
  /usr/src/app/frontend # npx expo start
  ```

  * react-typescript : `npx create-expo-app app --template`


  - EC2 구동시
    - [해당 EC2] - [네트워킹] - [IPv4 방화벽 규칙] - [규칙 추가]
      - port 19000, 19001, 19002 추가
    - connect to `[EC2 public ip]:3000`


### Step 1-2. Docker Code

* **File structure**

  ```bash
  .
  ├── 📄docker-compose.yml
  └── 📁myapp
      ├── 📄Dockerfile
      ├── 📁...
      └── 📁...
  ```

* `bash`

  ```bash
  $ mv Dockerfile myapp/
  ```

* `docker-compose.yml`

  ```dockerfile
  version: '3.8'
    
  services:
    app:
      container_name: app
      build:
        context: ./myapp/
        dockerfile: Dockerfile
      volumes:
        - ${PWD}/myapp/:/usr/src/app/
      ports:
        - 3000:3000
      command: sh -c "npm install && npx expo start"
  ```

* `frontend/Dockerfile`

  ```dockerfile
  FROM node:18-alpine3.16
  
  WORKDIR /usr/src/app
  
  RUN npm install -g npm@latest
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


## STEP 3. TypeScript: Install Dependencies
