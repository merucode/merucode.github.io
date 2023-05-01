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
  *  [Dockerizing Django with Postgres, Gunicorn, and Nginx](https://testdriven.io/blog/dockerizing-django-with-postgres-gunicorn-and-nginx/)

* **Setting**

  * Windows - WSL(Ubuntu 20.04)

  * Docker desktop running

    * WSL(Ubuntu 20.04) setting
  * docker-engine(20.10.23) and docker-compose(2.15.1)

* You can find **final file structure** at last step

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
  FROM python:3.9-alpine # python 3.9 이미지를 베이스 이미지로 합니다
  
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
  ```

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
  SECRET_KEY = os.environ.get("SECRET_KEY")							## Update
  DEBUG = int(os.environ.get("DEBUG", default=0))						## Update
  ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS").split(" ") 	## Update
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
  $ docker exec db psql --username=mysite_user --dbname=mysite_dev
  
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

### Step 2-4. Check

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

## STEP 3. Gunicorn and Nginx(Deployment Environment)

### Step 3-1. Connect Gunicorn and Make env file for deployment

* **`./docker-compose.prod.yml`**(Create)

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
      command: gunicorn mysite.wsgi:application --bind 0.0.0.0:8000	# Update
      env_file:	
        - ./.env.prod		# Update
      depends_on:
        - db
    
    db:
      container_name: db
      image: postgres:12.0-alpine
      volumes:
        - postgres_data:/var/lib/postgresql/data/
      env_file:
        - ./.env.prod.db	  # Update
        
  volumes:
    postgres_data:
  ```

  * 운영 환경에서 더이상 필요하지 않기 떄문에 backend에서 volume 제거

* **`./.env.prod`**(Create)

  ```
  ### Django settings
  DEBUG=0
  SECRET_KEY=change_me
  DJANGO_ALLOWED_HOSTS=localhost 127.0.0.1 [::1]
  
  ### DB
  SQL_ENGINE=django.db.backends.postgresql
  SQL_DATABASE=mysite_prod
  SQL_USER=mysite_user
  SQL_PASSWORD=mysite_prod
  SQL_HOST=db
  SQL_PORT=5432
  
  ### entrypoint.sh
  DATABASE=postgres
  ```

* **`./.env.prod.db`**(Create)

  ```
  POSTGRES_USER=mysite_user
  POSTGRES_PASSWORD=mysite_prod
  POSTGRES_DB=mysite_prod
  ```

### Step 3-2. Check

* **`bash`**

  ```bash
  $ docker compose down -v
  $ docker compose -f docker-compose.prod.yml up -d --build
  # http://localhost:8000/admin
  # check connect well without static files
  
  $ docker exec db psql --username=mysite_user --dbname=mysite_prod
  $ docker volume ls
  $ docker volume inspect mysite_postgres_data
  # check db
  # if dev db volume not deleteed by 'docker compose down -v',
  # error occur when you migrate
  
  
  $ docker compose -f docker-compose.prod.yml down
  ```

### Step 3-3. Deployment Dockerfile

* **`./backend/django/entrypoint.prod.sh`**(create)

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
  
  exec "$@"
  ```

* **`bash`**(entrypoint.prod.sh 파일 권한 변경)

  ```bash
  ./backend/django$ chmod +x entrypoint.prod.sh
  ```

* **`./backend/django/Dockerfile.prod`**(create)

  ```dockerfile
  ###########
  # BUILDER #
  ###########
  
  # 공식 베이스 이미지를 pull
  FROM python:3.9-alpine as builder
  
  # 작업 공간설정
  WORKDIR /usr/src/app
  
  # 환경변수 설정
  ENV PYTHONDONTWRITEBYTECODE 1
  ENV PYTHONUNBUFFERED 1
  
  # psycopg2 디펜던시 설치
  RUN apk update \
      && apk add postgresql-dev gcc python3-dev musl-dev
  
  # 디펜던시 설치
  COPY ./requirements.txt .
  RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt
  
  #########
  # FINAL #
  #########
  
  # 공식 베이스 이미지를 pull
  FROM python:3.9-alpine
  
  # app user를 위한 폴더 생성
  RUN mkdir -p /home/app
  
  # app user 생성
  RUN addgroup -S app && adduser -S app -G app
  
  # 적절한 디렉토리 생성
  ENV HOME=/home/app
  ENV APP_HOME=/home/app/web
  RUN mkdir $APP_HOME
  RUN mkdir $APP_HOME/staticfiles
  RUN mkdir $APP_HOME/mediafiles
  WORKDIR $APP_HOME
  
  # 디펜던시 설치
  RUN apk update && apk add libpq
  COPY --from=builder /usr/src/app/wheels /wheels
  COPY --from=builder /usr/src/app/requirements.txt .
  RUN pip install --no-cache /wheels/*
  
  # entrypoint-prod.sh 복사
  COPY ./entrypoint.prod.sh $APP_HOME
  
  # 프로젝트 파일 복사
  COPY . $APP_HOME
  
  # app user 모든 파일 권한변경
  RUN chown -R app:app $APP_HOME
  
  # app user 변경
  USER app
  
  # entrypoint.prod.sh 실행
  ENTRYPOINT ["/home/app/web/entrypoint.prod.sh"]
  ```

  * 최종 이미지 사이즈를 줄이기위해 multi-stage 빌드 도커를 사용
  * root 가 아닌 유저를 생성(보안)

* **`docker-compose.prod.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    backend:
      container_name: backend
      build:
        context: ./backend/django/
        dockerfile: Dockerfile.prod	# Update
  ...
  ```

### Step 3-4. Check

* **`bash`**

  ```bash
  $ docker compose -f docker-compose.prod.yml down -v
  $ docker compose -f docker-compose.prod.yml up -d --build
  # http://localhost:8000/admin
  # check connect well without static files
  
  $ docker compose -f docker-compose.prod.yml exec web python manage.py migrate --noinput
  # check migrate well
  
  $ docker compose -f docker-compose.prod.yml down -v
  ```

### Step 3-5. Nginx

* **`docker-compose.prod.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    backend:
      container_name: backend
      build:
        context: ./backend/django/
        dockerfile: Dockerfile.prod
      expose:			# Update
        - 8000		# Update	
      command: gunicorn mysite.wsgi:application --bind 0.0.0.0:8000
      env_file:
        - ./.env.prod
      depends_on:
        - db
    
    nginx:					# Add
      container_name: nginx
      build:
        context: ./backend/nginx/
        dockerfile: Dockerfile
      ports:
        - 80:80
      depends_on:
        - backend
  
    db:
      container_name: db
      ...
  ```

* **`./backend/nginx/Dockerfile`**(create)

  ```dockerfile
  FROM nginx:1.19.0-alpine
  
  RUN rm /etc/nginx/conf.d/default.conf
  COPY nginx.conf /etc/nginx/conf.d
  ```

* **`./backend/nginx/nginx.conf`**(create)

  ```nginx
  upstream mysite {
      server backend:8000;
  }
  
  server {
  
      listen 80;
  
      location / {
          proxy_pass http://mysite;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header Host $host;
          proxy_redirect off;
      }
  }
  ```

### Step 3-6. Check

* **`bash`**

  ```bash
  $ docker compose -f docker-compose.prod.yml down -v
  $ docker compose -f docker-compose.prod.yml up -d --build
  # connect to 'http://localhost:80/admin'
  # check connect well without static files
  
  $ docker compose -f docker-compose.prod.yml exec web python manage.py migrate --noinput
  # check migrate well
  
  $ docker compose -f docker-compose.prod.yml logs -f
  # docker compose logs check possible
  # ctrl+c : quit
  
  $ docker compose -f docker-compose.prod.yml down -v
  ```
  



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 4. Static/Media files(Deployment Environment) 

### Step 4-1. Static files

* **`./backend/django/mysite/settings.py`**

  ```python
  ...
  STATIC_URL = "/staticfiles/"						# Update
  STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")	# Add
  ...
  ```

* **`docker-compose.prod.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    backend:
      container_name: backend
      build:
        context: ./backend/django/
        dockerfile: Dockerfile.prod
      volumes:
        - static_volume:/home/app/web/staticfiles # static 공유 디렉토리
      expose:
        - 8000
      command: gunicorn mysite.wsgi:application --bind 0.0.0.0:8000
      env_file:
        - ./.env.prod
      depends_on:
        - db
    
    nginx:
      container_name: nginx
      build:
        context: ./backend/nginx/
        dockerfile: Dockerfile
      volumes:
        - static_volume:/home/app/web/staticfiles # static 공유 디렉토리
      ports:
        - 80:80
      depends_on:
        - backend
  
    db:
      container_name: db
      image: postgres:12.0-alpine
      volumes:
        - postgres_data:/var/lib/postgresql/data/
      env_file:
        - ./.env.prod.db
        
  volumes:
    postgres_data:
    static_volume:	# static 공유 디렉토리
  ```

* **`./backend/django/Dockerfile.prod`**(description)

  ```dockerfile
  ...
  # 적절한 디렉토리 생성
  ENV HOME=/home/app
  ENV APP_HOME=/home/app/web
  RUN mkdir $APP_HOME
  RUN mkdir $APP_HOME/staticfiles	# static 공유 디렉토리
  RUN mkdir $APP_HOME/mediafiles
  WORKDIR $APP_HOME
  ...
  ```

  * docker compose는 일반적으로 root 사용자로써 볼륨을 마운트하는데 현재 우리가 사용하고 있는 root 가 아닌 사용자인 경우, 권한문제가 발생해 collectstatic 명령어가 동작하지 않을 수 있음

    * 해당 이슈를 해결하려면 아래와 같은 방법을 사용(우리는 1번 사용)

      1. 도커파일 안에 폴더를 생성

      2. 마운트 된 폴더의 권한을 변경

* **`./backend/nginx/nginx.conf`**

  ```nginx
  upstream mysite {
      server backend:8000;
  }
  
  server {
  
      listen 80;
  
      location / {
          proxy_pass http://mysite;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header Host $host;
          proxy_redirect off;
      }
  
      # static 관련 추가 부분
      location /staticfiles/ {
          alias /home/app/web/staticfiles/;
      }
  }
  ```

  

### Step 4-2. Check

* **`bash`**

  ```bash
  $ docker compose down -v
  $ docker compose -f docker-compose.prod.yml down -v
  
  $ docker compose -f docker-compose.prod.yml up -d --build
  $ docker exec backend python manage.py migrate --noinput
  # Check migrate well
  
  $ docker exec backend python manage.py collectstatic --no-input --clear
  # connect to 'http://localhost:80/admin'
  # check connect well with static files
  
  $ docker compose -f docker-compose.prod.yml down -v
  ```

  

### Step 4-3. Media files

* **`./backend/django/settings.py`**

  ```python
  ...
  MEDIA_URL = "/mediafiles/"
  MEDIA_ROOT = os.path.join(BASE_DIR, "mediafiles")
  ...
  ```

* **`./backend/django/urls.py`**

  ```python
  from django.contrib import admin
  from django.urls import path
  from django.conf import settings			# Add
  from django.conf.urls.static import static	# Add
  
  urlpatterns = [
      path('admin/', admin.site.urls),
  ]
  
  if bool(settings.DEBUG):        	# Add for dev environment
      urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
  ```



### Step 4-4. Dev Environment

* **`bash`(check)**

  ```bash
  $ docker-compose up -d --build
  # should make folder './backend/django/mediafiles' and place example image
  # check to connect with `http://localhost:8000/mediafiles/IMAGE_FILE`
  
  $ docker-compose down -v
  ```



### Step 4-5. Prod Environment

* **`docker-compose.prod.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    backend:
      container_name: backend
      build:
        context: ./backend/django/
        dockerfile: Dockerfile.prod
      volumes:
        - static_volume:/home/app/web/staticfiles # static 공유 디렉토리
        - media_volume:/home/app/web/mediafiles   # media 공유 디렉토리	# Add
      expose:
        - 8000
      command: gunicorn mysite.wsgi:application --bind 0.0.0.0:8000
      env_file:
        - ./.env.prod
      depends_on:
        - db
    
    nginx:
      container_name: nginx
      build:
        context: ./backend/nginx/
        dockerfile: Dockerfile
      volumes:
        - static_volume:/home/app/web/staticfiles # static 공유 디렉토리
        - media_volume:/home/app/web/mediafiles   # media 공유 디렉토리	# Add
      ports:
        - 80:80
      depends_on:
        - backend
  
    db:
      container_name: db
      image: postgres:12.0-alpine
      volumes:
        - postgres_data:/var/lib/postgresql/data/
      env_file:
        - ./.env.prod.db
        
  volumes:
    postgres_data:
    static_volume:
    media_volume:	   # Add
  ```

* **`./backend/django/Dockerfile.prod`**(description)

  ```dockerfile
  ...
  # 적절한 디렉토리 생성
  ENV HOME=/home/app
  ENV APP_HOME=/home/app/web
  RUN mkdir $APP_HOME
  RUN mkdir $APP_HOME/staticfiles	# staticfiles
  RUN mkdir $APP_HOME/mediafiles	# mediafiles
  WORKDIR $APP_HOME
  ...
  ```

* **`./backend/nginx/nginx.conf`**

  ```nginx
  upstream hello_django {
      server backend:8000;
  }
  
  server {
  
      ...
  
      # media 관련 추가 부분
      location /mediafiles/ {
          alias /home/app/web/mediafiles/;
      }
  }
  ```

* **`bach`(check)**

  ```bash
  $ docker compose down -v
  $ docker compose -f docker-compose.prod.yml up -d --build
  $ docker exec backend manage.py migrate --noinput
  $ docker exec backend python manage.py collectstatics --noinput --clear
  # connect to `http://localhost:80/mediafiles/IMAGE_FILE`
  
  $ docker compose -f docker-compose.prod.yml down -v
  ```

    * 혹시 413 Request Entity Too Large 에러를 만나게된다면 Nginx 설정에서 클라이언트 request body 에 허용되는 최대파일의 크기를 변경

      ```nginx
      location / {
          proxy_pass http://hello_django;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header Host $host;
          proxy_redirect off;
          client_max_body_size 100M; # 추가된 부분
      }
      ```



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 5. Mid Check 

* **File Structure**

  ```
  .
  ├── 📁backend
  │   ├── 📁django
  │   │   ├── 📁mediafiles
  │   │   ├── 📁mysite
  │   │   ├── 📁staticfiles
  │   │   ├── 📄Dockerfile
  │   │   ├── 📄Dockerfile.prod
  │   │   ├── 📄entrypoint.prod.sh
  │   │   ├── 📄entrypoint.sh
  │   │   ├── 📄manage.py
  │   │   └── 📄requirements.txt
  │   └── nginx
  │       ├── 📄Dockerfile
  │       └── 📄nginx.conf
  ├── 📄docker-compose.prod.yml
  ├── 📄docker-compose.yml
  ├── 📄.env.dev
  ├── 📄.env.prod
  └── 📄.env.prod.db
  ```



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 6. AWS Connect

### Step 6-1. AWS Lightsail 인스턴스/DB 생성 및 네트워킹 설정

- **인스턴스 생성**
  - [운영체제] : Ubuntu 20.04 LTS
  - 고정 IP 생성
    - [네트워킹] : 고정 IP 생성, 인스턴스에 연결
  - 방화벽 규칙 설정
    - [네트워킹] - [IPv4 방화벽] - [규칙 추가]
    - [포트 또는 범위/코드] : 80
  - 참고사이트 : [점프 투 장고](https://wikidocs.net/164361)

* **DB 생성**
  * [데이터베이스] : PostgreSQL
  * [네트워킹] : 퍼블릭 모드
  * 참고사이트 : [점프 투 장고](https://wikidocs.net/75561)



### Step 6-2. Connect django AWS DB(Prod Environment)

* **`docker-compose.prod.yml`**

  ```dockerfile
  version: '3.7'
  
  services:
    backend:
      container_name: backend
      build:
        context: ./backend/django/
        dockerfile: Dockerfile.prod
      volumes:
        - static_volume:/home/app/web/staticfiles # static 공유 디렉토리
        - media_volume:/home/app/web/mediafiles   # media 공유 디렉토리
      expose:
        - 8000
      command: gunicorn mysite.wsgi:application --bind 0.0.0.0:8000
      env_file:
        - ./.env.prod
    
    nginx:
      container_name: nginx
      build:
        context: ./backend/nginx/
        dockerfile: Dockerfile
      volumes:
        - static_volume:/home/app/web/staticfiles # static 공유 디렉토리
        - media_volume:/home/app/web/mediafiles   # media 공유 디렉토리
      ports:
        - 80:80
      depends_on:
        - backend
        
  volumes:
    static_volume:
    media_volume:
  ```

* **`.env.prod`**

  ```python
  ### Django settings
  DEBUG=0
  SECRET_KEY=change_me
  DJANGO_ALLOWED_HOSTS=localhost 127.0.0.1 [::1]
  
  ### DB
  SQL_ENGINE=django.db.backends.postgresql
  SQL_DATABASE=	#aws db 
  SQL_USER=		#aws db user
  SQL_PASSWORD=	#aws db password
  SQL_HOST=		#endpoint of aws db
  SQL_PORT=5432
  
  ### entrypoint.sh
  DATABASE=postgres
  ```

* **`.env.prod.db` delete**

* **`bash`(check)**

  ```bash
  $ docker compose -f docker-compose.prod.yml up -d --build
  # connect to 'http://localhost/80/admin'
  
  $ docker compose -f docker-compose.prod.yml down -v
  ```



### Step 6-3. Git Push Project

* When you push to git hub, you should remove `.env.prod.db` or add `.gitignore` `.env.prod.db` for security



### Step 6-4. 인스턴스 docker/docker compose install

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

* **`bash`(sudo 없이 사용 가능하게 docker 권한 부여)** 

  ```bash
  $ sudo usermod -aG docker $USER
  # OR
  $ sudo usermod -aG docker $(whoami)
  
  # console 종료 후 재연결
  ```




## STEP 6-5. 인스턴스 github pull 및 runserver

* 인스턴스 git pull django project 

  * make `.env.prod`

* **`bash`(check)**

  ```bash
  $ docker compose -f docker-compose.prod.yml up -d --build
  $ docker exec backend python manage.py migrate --noinput
  $ docker exec backend python manage.py collectstatic --no-input --clear
  
  # connect to 'http://3.38.135.129/admin'
  $ docker compose -f docker-compose.prod.yml down
  ```
