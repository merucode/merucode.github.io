---
layout: default
title: React with Docker
parent: React
grand_parent: Frontend
nav_order: 7
---

# React with Docker
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

* 참고사이트 : [프론트엔드 개발자를 위한 Docker로 React 개발 및 배포하기](https://velog.io/@oneook/Docker%EB%A1%9C-React-%EA%B0%9C%EB%B0%9C-%EB%B0%8F-%EB%B0%B0%ED%8F%AC%ED%95%98%EA%B8%B0)



## STEP 1. Create React App

- **`filestructure`**

  ```
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
  RUN npm install -g create-react-app
  ```

- **`docker-compose.yml`**

  ```dockerfile
  version: '3'
    
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
      command: /bin/sh   # run -it mode
      stdin_open: true   # run -it mode
      tty: true		   # run -it mode
  ```

- **`bash`(create react  app and dev server run)**

  ```bash
  $ docker compose up -d --build
  $ docker exec -it frontend /bin/sh
  /usr/src/app # create-react-app myapp
  
  /usr/src/app # cd myapp
  /usr/src/app/myapp # npm run start
  ```

  - you can connent to http://localhost:3000

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. Install Dependencies

- **`bash`**

```bash
$ docker compose up -d --build
$ docker exec -it frontend /bin/sh
 /usr/src/app # cd myapp
 /usr/src/app/myapp # npm install react-router-dom@6
 /usr/src/app/myapp # npm install styled-components@5
```


<br>



<!------------------------------------ STEP ------------------------------------>
