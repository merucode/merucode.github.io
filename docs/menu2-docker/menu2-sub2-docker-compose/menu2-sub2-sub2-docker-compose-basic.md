---
layout: default
title: Doceker-compose Basic
parent: Docker-compose
grand_parent: Docker
nav_order: 2
---

# Docker-compose Basic
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

### Step. docker-compose에서 Dockerfile -it 모드로 run 하기

* **`docker-compose.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    app:
      container_name: myapp
      build:
        context: ./app/
        dockerfile: Dockerfile.app
      volumes:
        - ${PWD}/app/:/usr/src/app
      ports:
        - 80:80
  
      ### ubuntu
      command: /bin/bash   # run -it mode
      stdin_open: true	 # run -it mode
      tty: true			 # run -it mode
  
  	### alpine
      command: /bin/sh  	 # run -it mode
      stdin_open: true	 # run -it mode
      tty: true			 # run -it mode
  ```

* **`bash`**

  ```bash
  $ docker compose up -d --build
  
  ### ubuntu
  $ docker exec -it myapp /bin/bash
  
  ### alpine
  $ docker exec -it myapp /bin/sh
  ```

