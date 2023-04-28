---
layout: default
title: Deploy with Docker
parent: Django
grand_parent: Backend
nav_order: 8
---

# Deploy with Docker
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

## STEP 0. Reference

* **Reference Site**

  *  [Django 그리고 Postgres, Gunicorn, Nginx Dockerizing](https://parksunwoo.github.io/docker/2021/05/29/django-postgres-gunicorn-nginx-dockerizing.html)

* **Setting**

  * Windows - WSL(Ubuntu 20.04)

  * Docker desktop running

    * WSL(Ubuntu 20.04) setting

  
    ​	
  

<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 1. Make File Structure

## Step 1-1. Make File Structure

* **Basic File Structure**

  ```
  📁mysite
  ├── 📁backend
  │   ├── 📁django
  │   │   └── 📄Dockerfile
  │   │   └── 📄requirements.txt
  │   └── 📁nginx
  └── 📄docker-compose.yml
  ```

* **`Dockerfile`**

  ```dockerfile
  # python 3.9 이미지를 베이스 이미지로 합니다
  FROM python:3.9-alpine
  
  # 작업용 디렉토리를 지정합니다
  WORKDIR /usr/src/app
  
  # 환경 변수를 설정합니다
  ENV PYTHONDONTWRITEBYTECODE 1
  ENV PYTHONUNBUFFERED 1
  
  # psycopg2 dependencies 설치합니다
  RUN apk update \
      && apk add postgresql-dev gcc python3-dev musl-dev
  
  # 패키지들을 설치합니다
  RUN pip install --upgrade pip
  COPY ./requirements.txt .
  RUN pip install -r requirements.txt
  
  # 호스트상의 프로젝트 파일들을 이미지 안에 복사합니다
  COPY . .

* **`requirements.txt`**

  ```
  Django==4.0
  gunicorn==20.0.4
  psycopg2-binary==2.8.5
  ```

* **`docker-compose.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    backend:
      container_name: backend
      build:
        context: ./backend/django/
        dockerfile: Dockerfile
      ports:
        - "8000:8000"
      volumes:
        - ./backend/django/:/usr/src/app/
  ```

### Step 1-2. Check

 * **`bash`**

   ```bash
   $ docker compose up -d --build
   
   # After build well
   $ docker compose down
   ```

### Step 1-3. Create Django Project

 * **`docker-compose.yml`**

   ```bash
   version: '3.7'
   
   services:
     backend:
       container_name: backend
       build:
         context: ./backend/django/
         dockerfile: Dockerfile
       ports:
         - 8000:8000
       volumes:
         - ./backend/django/:/usr/src/app/
       command: ["django-admin","startproject","mysite","."]	## Add
   ```

* **`bash`**

  ```bash
  $ docker compose up -d --build
  # After create project
  $ docker compose down
  ```

## Step 1-4. Check runserver

* **`docker-compose.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    backend:
      container_name: backend
      build:
        context: ./backend/django/
        dockerfile: Dockerfile
      ports:
        - 8000:8000
      volumes:
        - ./backend/django/:/usr/src/app/
      command: ["python","manage.py","runserver","--noreload","0:8000"]	## Update
  ```

* **`bash`**

  ```bash
  $ docker compose up -d --build
  # Chekc to connect with http://localhost:8000/
  $ docker compose down
  ```

  

<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 2. Connect Local Postgres(Dev)

### Step 2-1. Connect Local Postgres(Dev)

* **`docker-compose.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    backend:
      container_name: backend
      build:
        context: ./backend/django/
        dockerfile: Dockerfile
      ports:
        - 8000:8000
      volumes:
        - ./backend/django/:/usr/src/app/
      command: ["python","manage.py","runserver","--noreload","0:8000"]
      env_file:
        - ./.env.dev
      depends_on:
        - db
    
    db:
      container_name: db
      image: postgres:12.0-alpine
      volumes:
        - postgres_data:/var/lib/postgresql/data/
      environment:
        - POSTGRES_USER=mysite_user
        - POSTGRES_PASSWORD=mysite
        - POSTGRES_DB=mysite_dev
        
  volumes:
    postgres_data:
  ```

* **`./.env.dev`**(Create)

  ```
  DEBUG=1
  SECRET_KEY=foo
  DJANGO_ALLOWED_HOSTS=localhost 127.0.0.1 [::1]
  
  SQL_ENGINE=django.db.backends.postgresql
  SQL_DATABASE=mysite_dev
  SQL_USER=mysite_user
  SQL_PASSWORD=mysite
  SQL_HOST=db
  SQL_PORT=5432
  ```

* **`./backend/django/mysite/settings.py`**

  ```python
  import os	## Add
  
  ...
  
  ## Update
  DATABASES = {
      "default": {
          "ENGINE": os.environ.get("SQL_ENGINE", "django.db.backends.sqlite3"),
          "NAME": os.environ.get("SQL_DATABASE", os.path.join(BASE_DIR, "db.sqlite3")),
          "USER": os.environ.get("SQL_USER", "user"),
          "PASSWORD": os.environ.get("SQL_PASSWORD", "password"),
          "HOST": os.environ.get("SQL_HOST", "localhost"),
          "PORT": os.environ.get("SQL_PORT", "5432"),
      }
  }
  
  ...
  ```
  

### Step 2-2. Check

* **`bash`**

  ```bash
  $ docker-compose up -d --build
  $ docker docker exec backend python manage.py migrate --noinput
  
  # After migrate complete
  $ docker-compose down -v
  ```

* **`bash`**

  ```bash
  $ docker-compose up -d --build
  
  # Create db
  $ docke exec db psql --username=mysite_user --dbname=mysite_dev
  
  # Check db
  $ docker volume ls
  $ docker volume inspect mysite_postgres_data
  ```

### Step 2-3. Make entrypoint.sh(Check DB and migrate)

* **`./backend/django/entrypoint.sh`**

  ```sh
  #!/bin/sh
  
  if [ "$DATABASE" = "postgres" ]
  then
      echo "Waiting for postgres..."
  
      while ! nc -z $SQL_HOST $SQL_PORT; do
        sleep 0.1
      done
  
      echo "PostgreSQL started"
  fi
  
  python manage.py flush --no-input
  python manage.py migrate
  
  exec "$@"
  ```

* **`bash`**(entrypoint.sh 파일 권한 변경)

  ```bash
  ./backend/django$ chmod +x entrypoint.sh
  ```

* **`Dockerfile`**

  ```dockerfile
  # 공식 베이지 이미지를 pull 합니다
  FROM python:3.8.3-alpine
  
  # 작업공간을 설정합니다
  WORKDIR /usr/src/app
  
  # 환경 변수를 설정합니다
  ENV PYTHONDONTWRITEBYTECODE 1
  ENV PYTHONUNBUFFERED 1
  
  # psycopg2 dependencies 설치합니다
  RUN apk update \
      && apk add postgresql-dev gcc python3-dev musl-dev
  
  # requirements 를 설치합니다
  RUN pip install --upgrade pip
  COPY ./requirements.txt .
  RUN pip install -r requirements.txt
  
  # 프로젝트 소스를 복사합니다
  COPY . .
  
  # run entrypoint.sh
  ENTRYPOINT ["/usr/src/app/entrypoint.sh"]	## Add
  ```

* **`.env.dev `**

  ```
  ### Django settings
  DEBUG=1
  SECRET_KEY=foo
  DJANGO_ALLOWED_HOSTS=localhost 127.0.0.1 [::1]
  
  ### DB
  SQL_ENGINE=django.db.backends.postgresql
  SQL_DATABASE=mysite_dev
  SQL_USER=mysite_user
  SQL_PASSWORD=mysite
  SQL_HOST=db
  SQL_PORT=5432
  
  ### entrypoint.sh
  DATABASE=postgres	## Add
  ```

## Step 2-4. Check

* **`bash`**

  ```bash
  $ docker-compose up -d --build
  $ docker logs backend
  
  # Check echo and migrate
  $ docker-compose down -v
  ```

### Step 2-5. Note

* **`entrypoint.sh`** 명렁어 주석 처리 및 명령어 직접 실행 가능

  ```sh
  #!/bin/sh
  
  if [ "$DATABASE" = "postgres" ]
  then
      echo "Waiting for postgres..."
  
      while ! nc -z $SQL_HOST $SQL_PORT; do
        sleep 0.1
      done
  
      echo "PostgreSQL started"
  fi
  
  # python manage.py flush --no-input
  # python manage.py migrate
  
  exec "$@"
  ```

* **`bash`**

  ```bash
  $ docker exec backend python manage.py flush --no-input
  $ docker exec backend python manage.py migrate
  ```



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 3.









































<!------------------------------------ STEP ------------------------------------>

## STEP 2. Dockerfile and docker-compose.yml 생성

* `Dockerfile`

  ```dockerfile
  FROM python:3.9-alpine
  
  WORKDIR /code
  
  # WORKDIR에 코드 복사 및 패키지 설치
  COPY . .
  RUN pip install -r requirements.txt
  
  # 개발 서버 실행(container 8000 접속에 대하여)
  CMD ["python","manage.py","runserver","--noreload","0:8000"]
  ```

* `docker-compose.yml`

  ```dockerfile
  version: '3'
  
  services:
    backend:
  		# backend dockerfile build
      build:
        context: ./backend
        dockerfile: Dockerfile
  		# host 8000 접속 시 container 8000 접속
      ports:
        - "8000:8000"
  		
      # 나중에 volumes 추가 예정 코드 임시 저장
      # volumes:
      #   - ./backend:/code
  ```



<br>



<!------------------------------------ STEP ------------------------------------>

##  STEP  3. AWS Lightsail 인스턴스 생성 및 네트워킹 설정

- 인스턴스 생성
  - [운영체제] : Ubuntu 20.04 LTS
- 고정 IP 생성
  - [네트워킹] : 고정 IP 생성, 인스턴스에 연결
- 방화벽 규칙 설정
  - [네트워킹] - [IPv4 방화벽] - [규칙 추가]
    - [포트 또는 범위/코드] : 8000
- 참고사이트: [점프 투 장고](https://wikidocs.net/164361)



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 4. 인스턴스 docker/docker compose install

* docker-desktop은 docker-engine과 windows 혹은 mac을 연결해주는 프로그램

  * 서버는 ubuntu만 이용하므로 docker-engine으로 충분

* `bash`(**docker/docker compose 설치**)

  ```bash
  ### Set up the repository
  #1. Update the apt package index and install packages to allow apt to use a repository over HTTPS:
  sudo apt-get update
  sudo apt-get install \
      ca-certificates \
      curl \
      gnupg
  
  #2. Add Docker’s official GPG key:
  sudo mkdir -m 0755 -p /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  
  #3. Use the following command to set up the repository:
  echo \
    "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
  ### Install Docker Engine
  #1. Update the apt package index:
  sudo apt-get update
  
  #2. Install Docker Engine, containerd, and Docker Compose.
  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  
  #3. docker install check :  --rm명령어를 주어 컨테이너 실행 후 바로 삭제되도록 한다.
  sudo docker run --rm hello-world
  
  ### Install Docker-compose
  #1. Update the package index, and install the latest version of Docker Compose:
  sudo apt-get update
  sudo apt-get install docker-compose-plugin
  
  #2. Verify that Docker Compose is installed correctly by checking the version.
  docker compose version
  ```

  - [docker install(linux)](https://docs.docker.com/desktop/install/ubuntu/)
  - [docker compose install(linux)](https://docs.docker.com/compose/install/linux/#install-using-the-repository)



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 5. 인스턴스 github pull 및 runserver

* 인스턴스 git pull django project 

* `bash`(runserver)

  ```bash
  $ sudo docker compose up
  ```

- 사이트 접속
  - url : `http://인스턴스외부고정IP:8000`
    - django 개발서버 https 지원 안됨



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 6. Dockfile Format 사용으로 배포

* git pull
* 
