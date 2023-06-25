---
layout: default
title: PostgreSQL
parent: Docker Format
grand_parent: Docker
nav_order: 7
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

## STEP 0. Reference Site

* [Github](https://github.com/merucode/form/tree/postgresql_basic)

<br>

## STEP 1. Docker Code(For DEV)

### Step 1-1. File Structure

* **File structure**

  ```bash
  .
  ├── 📁database
  │   ├── 📄Dockerfile
  │   └── 📁postgresql
  └── 📄docker-compose.yml
  ```

### Step 1-2. Docker Code

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
        - .database.env
  ```

* `database/Dockerfile`

  ```dockerfile
  FROM postgres:14.8
  ```

* `.database.env`

  ```bash
  POSTGRES_PASSWORD=test_password
  POSTGRES_USER=test_user
  POSTGRES_DB=test_db
  ```

  * 최초 실행 시 POSTGRES_USER, PASSWORD, DB로 유저, DB생성(Initalize)
    * POSTGRES_PASSWORD는 꼭 환경 변수 설정해줘야 접속 가능
  * 이후에는 해당 환경 변수들은 불필요

<br>

## STEP 2. Advance

### Step 2-1. Postgresql 접속 방법

* `bash`

  ```bash
  $ docker compose up -d --build
  $ docker exec -it database /bin/bash
  > su - postgres                            # user postgres 변경
  > psql -U [POSTGRES_USER] -d [POSTGRES_DB] # psql 접속 # psql -U test_user -d test_db;
  ```

### Step 2-2. Github 올릴시 기존 데이터들 저장 안됨 관련

* `database/postgres/data` 하부 빈 폴더들에 `.gitkeep` 생성 후 git push 하면 기존 데이터들도 저장될 것 같으나, 개발용으로만 주로 사용되기에 불필요하다고 생각되서 미수행

* `bash`(`cd postgres` 권한 문제 발생 해결 방법)

  ```bash
  $ sudo chown -R $(whoami) .
  ```