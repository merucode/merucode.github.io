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

### Step 2-1. Read Data

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

* **4. Check**

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



### Step 2-2. Create Data

* **Exectuion Step**

  1. Modify serializers
  2. Modify views
  3. Check

* **Modify serializers**

  * `movies/serializers.py`

    ```python
    class MovieSerializer(serializers.Serializer):
        id = serializers.IntegerField(read_only=True)		# Update 
        name = serializers.CharField()
        opening_date = serializers.DateField()
        running_time = serializers.IntegerField()
        overview = serializers.CharField()
    
        def create(self, validated_data):					### Add
            return Movie.objects.create(**validated_data)	###
    ```

    * `id` 필드에 `read_only`라는 옵션이 사용. 필드를 조회 시에만 사용하고 싶을 때 쓰는 옵션
    * `create()` 함수는 파라미터로 `validated_data`를 받습니다. `validated_data`는 유효성 검사를 마친 데이터라는 의미로, `MovieSerializer` 필드들에 해당하는 데이터가 딕셔너리 형태로 전달
    * `Movie` 모델에 `Movie.objects.create()`로 `validated_data`를 넣어 주면 데이터가 생성됩니다. 이때, 언패킹(`**`)을 사용하면 쉽게 처리
      * 언패킹을 사용하지 않고, 직접 입력하는 경우 : name, opening_data 등 모두 별도 입력 필요
      * 언패킹을 사용하는 경우 : 딕셔너리 형태의 변수 하나로 언패킹하여 입력 가능

* **Modify views**

  * `movies/views.py`

    ```python
    ...
    from rest_framework import status	# 상태 응답을 위한 라이브러리
    ...
    
    @api_view(['GET', 'POST'])			# POST 요청도 받을 수 있도록 POST 추가
    def movie_list(request):
        if request.method == 'GET':		# GET 요청일 경우 이전 데이터 조회 처리
            movies = Movie.objects.all()
            serializer = MovieSerializer(movies, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        elif request.method == 'POST':	# POST 요청일 경우 데이터 생성 처리
            data = request.data
            serializer = MovieSerializer(data=data)
            if serializer.is_valid():	# 유효성 검사
                serializer.save()		# save() 함수를 통한 create() 함수 실행
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    ```

    * `save()` 함수가 실행되어 `MovieSerializer`에서 정의했던 `create()` 함수가 실행되고, `Movie` 객체가 생성

* **Check**

  * `bash`

    ```bash
    docker compose up -d --build
    ```

  * Connect to `http://localhost:8000/movies`

  * 페이지 내 Content 박스에 아래 내용 입력 후 POST 버튼 눌러서 요청

    ```
    {
        "name": "반도",
        "opening_date": "2020-07-15",
        "running_time": 115,
        "overview": "전대미문의 재난 이후 4년이 흐른 대한민국은 버려진 땅이 되어버렸다. 사람들은 고립된 섬이 된 반도에 갇혔고 누구의 생사도 확인할 수 없는 상황에서 정석은 피할 수 없는 미션을 받고 한국 땅에 다시 발을 들인다. 정석은 미지의 세계인 그곳에서 예상치 못한 습격을 받고 일촉즉발의 순간 ‘반도’의 생존자들을 만나게 된다."
    }
    ```




### Step 2-3. Read, Update, Delete Specific Data

* **What is End point?**

  * End point: API를 생성할 때 엔드 포인트란 서버의 리소스(데이터)에 접근하게 해주는 URL

  * End point for specific data

    | HTTP Method | 엔드포인트  | 기능                         | CRUD           |
    | ----------- | ----------- | ---------------------------- | -------------- |
    | GET         | /movies/:id | 특정한 영화 데이터 조회      | Read           |
    | PATCH       | /movies/:id | 특정한 영화 데이터 부분 수정 | Partial Update |
    | PUT         | /movies/:id | 특정한 영화 데이터 모두 수정 | Update         |
    | DELETE      | /movies/:id | 특정한 영화 데이터 삭제      | Delete         |

* **Exectuion Step**

  1. Setting URLs
  2. Connect to endpoint
  3. Modify serializer(for data update, delete)
  4. Modify view(for data update, delete)
  5. Check

* **Setting URLs**

  * `movies/urls.py`

    ```python
    from django.urls import path
    from .views import movie_list, movie_detail
    
    urlpatterns = [
        path('movies', movie_list), 
        path('movies/<int:pk>', movie_detail),    
    ]
    ```

    * 영화의 `id`는 URL 파라미터로 받으면 되는데요. 이때 `pk`라는 이름으로 파라미터가 전달됩니다(`pk`는 Primary key, 즉 `id`를 뜻합니다).

* **Connect to endpoint**

  * `movies/views.py`

    ```python
    @api_view(['GET', 'PATCH', 'DELETE'])
    def movie_detail(request, pk):	
        pass
    ```

    * `urls.py`에 작성한 변수명인 `pk`를 파라미터로 넘겨주면 영화를 구분할 수 있는 식별자를 뷰에서 사용

* **Modify serializer(for data update, delete)**

  * `movies/serializers.py`

    ```python
    class MovieSerializer(serializers.Serializer):
        ...
        
        def create(self, validated_data):
            ...
            
        def update(self, instance, validated_data):
            instance.name = validated_data.get('name', instance.name)
            instance.opening_date = validated_data.get('opening_date', instance.opening_date)
            instance.running_time = validated_data.get('running_time', instance.running_time)
            instance.overview = validated_data.get('overview', instance.overview)
            instance.save()
            return instance
    ```

    * `validated_data`는 `create()` 함수에서와 마찬가지로 유효성 검사를 마친 데이터
    * `instance`는 수정할 데이터
    * 데이터를 수정하는 방식이 `PUT`(모든 필드의 데이터 수정)이 아니라 `PATCH`(특정 필드의 데이터 수정)이므로, 수정 요청이 들어온 필드만 `validated_data`로 수정하고, 나머지는 기존의 값을 그대로 사용
    * `get()`은 파라미터로 키(Key)와 기본값(Default Value)을 받습니다. 만약, 딕셔너리에 키에 맞는 데이터가 존재한다면 데이터를 반환하고, 키에 맞는 데이터가 존재하지 않다면 설정한 기본값을 반환
    * 이후, `Movie` 객체에 존재하는 `save()` 함수로 수정한 값을 저장하고, 수정된 객체를 반환

* **Modify view(for data update, delete)**

  * `movies/views.py`

    ```python
    ...
    from rest_framework.generics import get_object_or_404
    # CH 3-4. 개별 객체 요청 정상 구분 처리를 위한 라이브러리
    ...
    
    @api_view(['GET', 'POST'])
    def movie_list(request):
        ...
    
    @api_view(['GET', 'PATCH', 'DELETE'])                   
    def movie_detail(request, pk):
        movie = get_object_or_404(Movie, pk=pk)             
        # 데이터 없을 경우 404 에러 처리
        # 첫 번째 파라미터로는 조회할 모델을, 두 번째 파라미터로는 조회할 pk 값을 입력
        
        if request.method == 'GET':                         # GET 요청 처리하기
            serializer = MovieSerializer(movie) 
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        if request.method == 'PATCH':                       # PATCH 요청 처리하기
            serializer = MovieSerializer(movie, data=request.data, partial=True) 
            if serializer.is_valid():
                serializer.save()    
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
        if request.method == 'DELETE':                      # DELETE 요청 처리하기
            movie.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
    ```

    *  `@api_view()`에는 사용할 HTTP Method인 `GET`(특정한 영화 데이터 조회), `PATCH`(특정한 영화 데이터 수정), `DELETE`(특정한 영화 데이터 삭제)를 작성
    * 특정한 영화 객체 데이터 `get_object_or_404()`로 받아오기(존재하지 않는다면 404 에러)
    * `PATCH` 요청이 들어오면 `MovieSerializer`에 수정하려는 영화 객체(`movie`)를 넣어 주고, 수정할 데이터(`request.data`)를 `data` 옵션에 넣어줌
      * `PATCH`는 부분 데이터 수정이기 때문에 `partial` 옵션을 `True`로 했습니다. 모든 데이터를 수정하는 `PUT` 요청을 처리해야 한다면 `partial` 옵션을 적지 않아도 됨
      *  `is_valid()`로 검증합니다. 정상적으로 처리되면 `serializer.data`와 함께 상태 코드 `200`을 반환
    * `DELETE` 요청이 들어오면 `movie` 객체를 `delete()` 함수로 삭제합니다. 데이터가 삭제되면 반환할 데이터가 없기 때문에 상태 코드인 `204`만 반환

* **Check**

  * `bash`

    ```bash
    docker compose up -d --build
    ```

  * Connect to `http://localhost:8000/movies/2`

  * 페이지 내 Content 박스에 아래 내용 입력 후 PATCH 버튼 눌러서 요청

    ```
    {
        "name": "부당거래",
        "opening_date": "2010-10-28",
        "running_time": 119,
        "overview":"온 국민을 충격으로 몰아넣은 연쇄 살인 사건. 계속된 검거 실패로 대통령이 직접 사건에 개입하고, 수사 중 용의자가 사망하는 사고가 발생하자 경찰청은 마지막 카드를 꺼내든다. 가짜 범인을 만들어 사건을 종결 짓는 것. 사건의 담당인 광역수사대 최철기는 승진을 보장해주겠다는 상부의 조건을 받아들이고 사건에 뛰어들게 된다. 그는 스폰서인 해동 장석구를 이용해 배우를 세우고 대국민을 상대로 한 이벤트를 완벽하게 마무리 짓는다. 한편 부동산 업계의 큰 손 태경 김회장으로부터 스폰을 받는 검사 주양은 최철기가 입찰 비리건으로 김회장을 구속시켰다는 사실에 분개해 그의 뒤를 캐기 시작하는데..."
    }
    ```

  * 페이지 내 DELETE 버튼 눌러서 삭제 요청

  * `http://localhost:8000/movies/2` 재접속 시 404 에러 확인




### Step 2-4. 

