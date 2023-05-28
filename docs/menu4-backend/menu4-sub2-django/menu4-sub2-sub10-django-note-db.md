---
layout: default
title: Django Note DB
parent: Django
grand_parent: Backend
nav_order: 10
---

# Django Note DB
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

## STEP 1. Use PostgreSQL

### Step 1-1. Install PostgreSQL

* `bash`
	```bash
	$ sudo apt update
	$ sudo apt install postgresql
	
	$ sudo service postgresql start	# 서버 시작
	$ sudo service postgresql stop	# 서버 종료
	$ sudo service postgresql restart # 서버 재시작
	```

### Step 1-2. Create Database

* 자동으로 `postgres`라는 유저와 `postgres`라는 데이터베이스를 생성
* `postgres` 유저는 모든 권한이 있는 유저

* `bash`
	```bash
	$ sudo service postgresql start 	# 서버시작
	$ sudo -u postgres psql 			# 서버 접속
	# 데이터베이스 목록 조회
	postgres=# \l 		
					
	# 데이터 베이스 생성
	postgres=# CREATE DATABASE dbname;

	# 새로운 유저 생성 및 권한 부여
	postgres=# CREATE USER username WITH PASSWORD 'password';
	postgres=# GRANT ALL PRIVILEGES ON DATABASE dbname TO username;
	
	# 커맨드 라인 종료
	postgres=# \q

	# 윈도우 WSL의 경우 DB 유저를 리눅스 시스템에 추가 필요
	$ sudo adduser username

	# username이라는 유저로 dbname 접속
	$ sudo -u username psql dbname

	# 데이터베이스 테이블 리스트 조회
	postgres=# \dt
	```

###  Step 1-3. Django Setting

* `bash`(install `psycopg2`)
	```bash
	$ sudo apt install python3-dev libpq-dev build-essential 
	$ pip install psycopg2
	```
	* `psycopg2`는 파이썬과 PostgresSQL 데이터베이스를 연결하는 인터페이스
* `settings.py`
	```python
	DATABASES = { 
	'default': { 
		'ENGINE': 'django.db.backends.postgresql', 
		'NAME': 'dbname', 
		'HOST': 'localhost', 
		'PORT': '5432', 
		'USER': 'username', 
		'PASSWORD': 'password', 
		} 
	}
	```
* `bash`
	```bash
	$ python manage.py migrate
	```

<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 2. Show Model Data

* `bash`

  ```bash
  python manage.py shell
  >> from ap  p_name.models import User, Review    # app의 User, Review 모델 import
  >> User.objects.all()                     # User 모델 객체 모든보기
  >> for user in User.objects.all():        # User 모델 칼럼 데이터 확인
        print(user.email, user.email_domain)  
  ```

<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 3. Show SQLite3 Data

### Step 3-1. VSCode "SQLite Viewer" Extension

### Step 3-2. Djagno Database Shell

* `bash`

  ```bash
  python manage.py dbshell
  sqlite> .tables           # Show all table list
  sqlite> .headers on       # 데이터를 조회할 때 컬럼에 대한 정보도 같이 조회
  sqlite> PRAGMA table_info('coplate_review');
  # 순서 | 컬럼이름 | 컬럼타입 | null여부 | 디폴트 | pk여부

  sqlite> SELECT * FROM coplate_user;
  # 테이블 조회

  sqlite> SELECT email, nickname FROM coplate_user;
  # 특정 컬럼 조회

  sqlite> SELECT email, nickname FROM coplate_user WHERE id=1;
  # 데이터 필터

  sqlte> .exit              
  ```

* 참고로 .으로 시작하는 커맨드는 SQLite3 전용 커맨드이기 때문에 다른 DBMS를 사용할 때는 사용할 수 없습니다.


