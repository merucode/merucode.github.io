---
layout: default
title: Doceker-compose Basic
parent: Docker-compose
grand_parent: Docker
nav_order: 1
---

# Docker-compose Basic
{: .no_toc }





### Step. Dockerfile run -it

```docker
  app:
    build:
      context: ./app/
      dockerfile: Dockerfile.app
    volumes:
      - ${PWD}/app/:/usr/src/app
    ports:
      - 80:80
    command: /bin/bash   # run -it mode
    stdin_open: true	 # run -it mode
    tty: true			 # run -it mode

```



