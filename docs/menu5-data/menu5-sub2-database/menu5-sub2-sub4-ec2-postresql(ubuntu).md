---
layout: default
title: EC2 Postgresql(Ubuntu)
parent: Database
grand_parent: Data
nav_order: 4
---

# EC2 Postgresql
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

## STEP 0. Reference Site

* [[DOC] postgresql](https://www.postgresql.org/download/linux/ubuntu/)
* [[BLOG] 우분투에 PostgreSQL 설치 및 접속 핵심 정리](https://backendcode.tistory.com/265)
* [[BLOG] [PostgreSQL 설치와 운영] #1. Ubuntu 22.04 PostgreSQL 설치](https://berasix.tistory.com/entry/PostgreSQL-Ubuntu-2204-PostgreSQL-%EC%84%A4%EC%B9%98)


<!------------------------------------ STEP ------------------------------------>

## STEP 1. Install Postgresql

* bash

```bash
# Create the file repository configuration:
sudo sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Import the repository signing key:
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# Update the package lists:
sudo apt-get update

# Install the latest version of PostgreSQL.
# If you want a specific version, use 'postgresql-12' or similar instead of 'postgresql':
sudo apt-get -y install postgresql
```

<!------------------------------------ STEP ------------------------------------>

## STEP 2. Setting Postgresql

### Step 2-1. Start Postgresql 

* bash

```bash
# postgresql 실행
$ sudo service postgresql start
 
# postgresql 상태 확인
$ sudo service postgresql status
 
# postgresql 종료
$ sudo service postgresql stop
```

## Step 2-2. Create User and Database

* bash

```bash
# postgresql 실행
$ sudo service postgresql start

# postgresql 접속
$ sudo -i -u postgres
$ psql

> ALTER USER postgres ENCRYPTED PASSWORD '[new_password]';	# postgres 비밀번호 설정

# create new user/db
> CREATE USER [new_user] NOSUPERUSER;
> ALTER USER [new_user] ENCRYPTED PASSWORD '[new_password]';
> CREATE DATABASE [new_db] WITH OWNER [new_user];
>\q
# GRANT ALL PRIVILEGES ON DATABASE [DB_NAME] TO [USER_NAME];
# [USER_NAME]에게 [DB_NAME] 전권 주기
```

### Step 2-3. PG 접속 설정 파일 수정

* bash

```bash
$ sudo vim /etc/postgresql/14/main/pg_hba.conf
$ sudo vim /etc/postgresql/14/main/postgresql.conf
```

* pg_hba.conf

```bash
  # 'local' is for Unix ...
  local all	   all                 md5  # 로컬 접속 시 peer 대신 md5 방식 사용
  # IPv4 local connections:
  ...
  host [new_db] [new_user] 0.0.0.0/0 md5 # ADD
```

* `postgresql.conf`

```
  ...
  listen_adresses = '*'	# 주석 해제 및 수정
  port = 5432 			# 주석 해제
  ...
```


### Step 2-4. 변경 설정 적용

* bash

```bash
# 변경 설정 적용을 위한 PG 재시작
$ sudo service postgresql restart
```

* EC2 Network 설정
* 퍼블릭 IP : 고정 IP 설정
* [EC2 인스턴스] → [네트워킹] → [IPv4 방화벽 규칙추가]
    * 어플리케이션 : PostgreSQL
    * 프로토콜 : TCP
    * 포트 : 5432
* bash(접속 test)

```bash
$ sudo -i -u postgres
$ psql # psql 접속
> \l   # DB 확인
> \du  # User 확인
> \q

$ psql -U [new_user] -d [new_db] # 로컬 DB 접속
> \q

$ psql -h 127.0.0.1 -U [new_user] -d [new_db]	# 로컬 ip 원격 db 접속
> \q

$ psql -h [ec2 ip] -U [new_user] -d [new_db]	# ec2 ip 원격 db 접속
```


<!------------------------------------ STEP ------------------------------------>

## STEP 3. DB Connection Info

```
Host: ec2 public ip
DB name: new_db  
Username: new_user  
password: new_password
port: 5432
```