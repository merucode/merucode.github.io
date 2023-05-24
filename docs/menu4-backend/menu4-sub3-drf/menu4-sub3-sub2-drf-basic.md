---
layout: default
title: DRF Basic
parent: DRF
grand_parent: Backend
nav_order: 9
---

# DRF Basic
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

## STEP 1. DRF(Django REST Framework)

### Step 1-1. What is DRF?

* Role of DRF

  ![image-20230524124007856](./../../../images/menu4-sub3-sub2-drf-basic/image-20230524124007856.png) 
  
  <img src="./../../../images/menu4-sub3-sub2-drf-basic/image-20230524124042525.png" alt="image-20230524124042525" style="zoom: 80%;" />

### Step 1-2. What is REST?

* REST(Representational State Transfer) : 자원을 이름으로 구분하여 해당 자원의 상태(정보)를 주고 받는 모든 것을 뜻

* Status Code

  | Status Code      | 항목                                                         |
  | ---------------- | ------------------------------------------------------------ |
  | 1XX (정보 전달)  | 100 (진행, Continue) / 101 (프로토콜 전환, Switching Protocol) |
  | 2XX (성공)       | 200 (성공, OK) / 201 (작성됨, Created) / 202 (허용됨, Accepted) / 203 (신뢰할 수 없는 정보, Non-Authoritative Information) /<br>204 (콘텐츠 없음, No Content) / 205 (콘텐츠 재설정, Reset Content) |
  | 3XX (리다이렉션) | 300 (여러 개의 선택 항목, Multiple Choice) / 301 (영구 이동, Moved Permanently) / 302 (임시 이동, Found) / <br>304 (수정되지 않음, Not Modified) |
  | 4XX (실패)       | 400 (잘못된 요청, Bad Request) / 401 (권한 없음, Unauthorized) / 402 (결제 필요, Payment Required) / <br>403 (금지됨, Forbidden) / 404 (찾을 수 없음, Not Found) |
  | 5XX (서버 오류)  | 500 (내부 서버 오류, Internal Server Error) / 501 (구현되지 않음, Not Implemented) / <br>502 (잘못된 게이트웨이, Bad Gateway) / 503 (서비스를 사용할 수 없음, Service Unavailable) |

* URL Desing

  | HTTP 메소드 | **설명**         | **URL 예시** |
  | ----------- | ---------------- | ------------ |
  | GET         | 조회             | /movies      |
  | POST        | 생성             | /movies      |
  | GET         | 특정 데이터 조회 | /movies/:id  |
  | PATCH       | 특정 데이터 수정 | /movies/:id  |
  | DELETE      | 특정 데이터 삭제 | /movies/:id  |

  ```
  http://localhost:8000/movies/1/ 		# X(마지막에 슬래시가 있는 것과 없는 것은 서로 다른 자원)
  URL: http://localhost:8000/movies/1 	# O
  
  http://localhost:8000/popular_movies	# X
  http://localhost:8000/PopularMovies		# X
  http://localhost:8000/popular-movies	# O(구분자 - 사용)
  ```



### Step 1-3. Preparing Development Environment(Install DRF)

* `Dockerfile`

  ```dockerfile
  FROM python:3.9-alpine 
  
  WORKDIR /usr/src/app
    
  ENV PYTHONDONTWRITEBYTECODE 1
  ENV PYTHONUNBUFFERED 1
  
  RUN pip install --upgrade pip
  COPY ./requirements.txt .
  RUN pip install -r requirements.txt
  
  COPY . .
  ```

* `docker-compose.yml`

  ```dockerfile
  version: '3.7'
    
  services:
    backend:
      container_name: drf
      build:
        context: ./
        dockerfile: Dockerfile
      ports:
        - "8000:8000"
      volumes:
        - ./:/usr/src/app/
      command: /bin/sh   # run -it mode
      stdin_open: true   # run -it mode
      tty: true		   # run -it mode
  ```

* `requirements.txt`

  ```
  Django==4.0
  djangorestframework==3.13.1
  ```

* `bash`

  ```bash
  $ docker compose up -d --build
  $ docker exec -it drf /bin/sh
  /usr/src/app # django-admin startproject movie_api
  /usr/src/app # exit
  $ docker compose down
  ```

* `movie_api/settings.py`

  ```python
  INSTALLED_APPS = [
      ...
      'django.contrib.staticfiles',
      'rest_framework'	# ADD
  ]
  ```

  * DRF를 사용하기 위해서는 `settings.py`에 있는 `INSTALLED_APPS`에 `rest_framework`를 추가



### Step 1-4. Preparing Development Environment(Create model)

* `bash`

  ```bash
  $ docker compose up -d --build
  $ docker exec -it drf /bin/sh
  /usr/src/app # cd movie_api/
  /usr/src/app/movie_api # python manage.py startapp movies
  /usr/src/app/movie_api # exit
  $ docker compose down
  ```

* `movie_api/settings.py`

  ```python
  INSTALLED_APPS = [
      ...
      'django.contrib.staticfiles',
      'rest_framework',
      'movies'	# ADD
  ]
  ```

* `movies/models.py`

  ```python
  from django.db import models
  
  
  class Movie(models.Model):					###  ADD
      name = models.CharField(max_length=30)
      opening_date = models.DateField()
      running_time = models.IntegerField()
      overview = models.TextField()			###
  ```

* [movies.json](https://bakey-api.codeit.kr/api/files/resource?root=static&seqId=5826&directory=movies.json&name=movies.json) 다운로드 및 manage.py 파일이 있는 폴더에 위치

* `bash`

  ```bash
  $ docker compose up -d --build
  $ docker exec -it drf /bin/sh
  /usr/src/app # cd movie_api/
  /usr/src/app/movie_api # python manage.py makemigrations
  /usr/src/app/movie_api # python manage.py migrate
  /usr/src/app/movie_api # python manage.py loaddata movies.json
  /usr/src/app/movie_api # exit
  $ docker compose down
  ```

  * `loaddata`는 JSON 형식의 파일로부터 데이터를 받아 Django 데이터 베이스에 입력해 주는 명령어



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 2. Serializer

### Step 2-1. Data lookup

* **Exectuion Step**

  1. Create serializers

  2. Connect between model and view
  3. Setting URL and check
  4. Check

* **1. Create Serializers**

  * `movies/serializers.py`(create)

    ```python
    from rest_framework import serializers	# 시리얼라이저를 생성하기 위한 라이브러리 
    from .models import Movie				# 사용할 모델
    
    class MovieSerializer(serializers.Serializer):
        id = serializers.IntegerField()
        name = serializers.CharField()
        opening_date = serializers.DateField()
        running_time = serializers.IntegerField()
        overview = serializers.CharField()
    ```

    *  사용할 필드 이름은 꼭 **모델에서 사용하는 필드 이름과 일치**

    * `id`는 Django 모델이 자동으로 정의해 주는 필드. `GET` 요청을 보낼 때 함께 조회하고 싶어서 추가

* **2. Connect between model and view**

  * `movies/views.py`

    ```python
    from rest_framework.decorators import api_view		# api 제어용(데코레이션)
    from rest_framework.response import Response		# 반환 시 JSON 변환용
    
    from .models import Movie							# 모델 데이터
    from .serializers import MovieSerializer			# 만들어 둔 Serializer 
    
    @api_view(['GET'])									# GET 메소드만 허용하는 API를 제공
    def movie_list(request):
        movies = Movie.objects.all()					# 모든 영화 객체를 가져와 
        serializer = MovieSerializer(movies, many=True)	
        # MovieSerializer에 입력 및 파이썬 딕셔너리 형태로 변환
        # 여러 데이터를 직렬화하는 것이라면 many=True 필요
        return Response(serializer.data, status=200)	# JSON 형태 변환
    ```

    * `@api_view(['GET'])`으로 함수형 뷰인 `movie_list()`가 `GET` 메소드만 허용하는 API를 제공
      * `movie_list()`를 수정하지 않고 `api_view()`의 기능을 추가

    * `MovieSerializer`에 파이썬 객체 형태의 데이터인 `movies`를 넣어 주면 데이터가 파이썬 딕셔너리 형태로 변환. 변환된 데이터에는 `serializer.data`로 접근

    * `Response`는 `rest_framework`에서 제공하는 특별한 응답 클래스
      * `MovieSerializer`를 통해 파이썬 딕셔너리로 변환된 데이터는 `Response`에서 최종적으로 JSON 형태로 변환

* **3. Setting URL**

  * `movie_api/urls.py`

    ```python
    from django.contrib import admin
    from django.urls import path, include
    
    urlpatterns = [
        path('admin/', admin.site.urls),
        path('', include('movies.urls')),
    ]
    ```

  * `movies/urls.py`

    ```python
    from django.urls import path
    from .views import movie_list
    
    urlpatterns = [
        path('movies', movie_list),    
    ]
    ```

* **4. Check **

  * `movie_api/Dockerfile`

    * Move Dockerfile to movie_api folder

  * `dockercompose.yml`

    ```
    version: '3.7'
      
    services:
      backend:
        container_name: drf
        build:
          context: ./movie_api/
          dockerfile: Dockerfile
        ports:
          - "8000:8000"
        volumes:
          - ./movie_api/:/usr/src/app/
        command: ["python","manage.py","runserver","--noreload","0:8000"]
    ```

  * `bash`

    ```bash
    docker compose up -d --build
    ```

  * Connect to `http://localhost:8000/movies`



