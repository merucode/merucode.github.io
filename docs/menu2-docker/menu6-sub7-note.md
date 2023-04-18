---
layout: default
title: Note
parent: Docker
nav_order: 7
---

# Note
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

## STEP 1. Background 실행 중인 python logs 출력방법

* python `-u` option 사용 후 `docker logs [container-id]`로 확인

  * `docker-compose.yml` : `command: python -u main.py`
  * `bash` : `docker logs [container-id]`(container 및 CMD 실행 중)
  
* **`docker-compose.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    app:
      container_name: myapp
      build:
        context: ./app/
        dockerfile: Dockerfile.prod
      volumes:
        - ${PWD}/app/:/usr/src/app
      env_file:
        - ./.env
      ports:
        - 80:80
      command: python -u main.py
  ```
  
* `bash`

  ```bash
  $ docker logs [container-id]
  ```

  

<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 2. Container 내부 실행 중인 CMD 확인

* `bash`

  ```bash
  $ docker exec [container-id] ps aux
  ```
  
* **Example**

  ```bash
  kym926151@crawling:~$ docker exec myapp ps aux
  USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
  root           1  2.5 11.5 230464 114580 ?       Ssl  10:42   5:04 python -u main.py
  root          14  0.0  0.3   8652  3084 ?        Rs   13:59   0:00 ps aux
  ```
  



<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 3. 실행 중인 Container 내부 코드 실행

* `bash`(container run in background)

  ```bash
  $ docker exec [container-id] [CMD]
  ```

* **Example**

  ```bash
  ```

  
