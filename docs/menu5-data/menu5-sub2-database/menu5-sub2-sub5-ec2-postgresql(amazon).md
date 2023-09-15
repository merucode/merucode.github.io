---
layout: default
title: EC2 Postgresql(Amazon)
parent: Database
grand_parent: Data
nav_order: 5
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

* [AWS EC2에 Postgresql 설치하기](https://velog.io/@jwpark06/AWS-EC2%EC%97%90-Postgresql-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0)
* [Install PostgreSQL on Amazon Linux](https://www.linuxage.net/2023/03/install-postgresql-on-amazon-linux.html)
* [pg_hbaconf-파일-method-옵션](https://docs.3rdeyesys.com/database/ncloud-database-postgresql-install-connect-guide-centos.html#pg_hbaconf-%ED%8C%8C%EC%9D%BC-method-%EC%98%B5%EC%85%98)

<!------------------------------------ STEP ------------------------------------>

## STEP 1. Install Posgresql(PG)

* Follow the steps given below to install the latest version of PostgreSQL on `Amazon Linux2` OS
Update yum cache and installed packages.
* `bash`

	```bash
	$ sudo yum update -y

	# PostgreSQL is part of the amazon extras library. 
	# Install the PostgreSQL amazon extras repository. 
	# At the time of writing, PostgreSQL 14 is the latest package available in the extras library.
	$ sudo amazon-linux-extras enable postgresql14
	
	# Install PostgreSQL server:
	$ sudo yum install -y postgresql-server

	# Initialize the DB:
	$ sudo postgresql-setup initdb

	# Add the PostgreSQL service to the system startup:
	$ sudo systemctl enable postgresql
	$ sudo systemctl start postgresql

	# Check the status of PostgreSQL using the following command:
	$ sudo systemctl status postgresql
	```

<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 2. Setting PG

### Step 2-1. Create User and DB

* `bash`

	```bash
	# 리눅스의 postgres(postgresql 설치시 생성) 유저로 전환(ec-user2는 postgres 권한 없음)
	# 리눅스 사용자 전환
	$ sudo su - postgres

	bash$ psql	# pg 접속 
	> ALTER USER postgres ENCRYPTED PASSWORD '[new_password]';	# postgres 비밀번호 설정

	# create new user/db
	> CREATE USER [new_user] NOSUPERUSER;
	> ALTER USER [new_user] ENCRYPTED PASSWORD '[new_password]';
	> CREATE DATABASE [new_db] WITH OWNER [new_user];
	>\q
	# GRANT ALL PRIVILEGES ON DATABASE [DB_NAME] TO [USER_NAME];
	# [USER_NAME]에게 [DB_NAME] 전권 주기
	```

### Step 2-2. PG 접속 설정 파일 수정

* `bash`

	```bash
	# ec2-user로는 지금 수정하려는 파일에 대한 접근이 불가능하기 때문에, 
	# `sudo su` 명령을 통해 root 유저로 접근
	$ sudo su - postgres
	bash$ cd /var/lib/pgsql/data/
	bash$ vim postgresql.conf
	```

* `postgresql.conf`

	```python
	...
	listen_adresses = '*'	# 주석 해제 및 수정
	port = 5432 			# 주석 해제
	...
	```

### Step 2-3.  PG 인증 설정 파일 수정

* `bash`

	```bash
	$ sudo su - postgres
	bash$ cd /var/lib/pgsql/data/
	bash$ vim pg_hba.conf
	```


* `pg_hba.conf`

	```python
	# 'local' is for Unix ...
	local all	   all                 md5
	# IPv4 local connections:
	...
	host [new_db] [new_user] 0.0.0.0/0 md5 # ADD
	```

### Step 2-4. 변경 설정 적용 및 test

* `bash`
	```bash
	# 변경 설정 적용을 위한 PG 재시작
	$ sudo systemctl restart postgresql
	```

* **EC2 Network 설정**
	* 퍼블릭 IP  : 고정 IP 설정
	* [EC2 인스턴스] → [네트워킹] → [IPv4 방화벽 규칙추가] 
		* 어플리케이션 : PostgreSQL
		*  프로토콜 : TCP 
		* 포트 : 5432 

* `bash`(접속 test)
	```bash
	$ sudo su - postgres
	bash$ psql # psql 접속
	> \l   # DB 확인
	> \du  # User 확인
	> \q
	
	bash$ psql -U [new_user] -d [new_db] # 로컬 DB 접속
	> \q
	
	bash$ psql -h 127.0.0.1 -U [new_user] -d [new_db]	# 로컬 ip 원격 db 접속
	> \q
	
	bash$ psql -h [ec2 ip] -U [new_user] -d [new_db]	# ec2 ip 원격 db 접속
	```

<br>

<!------------------------------------ STEP ------------------------------------>


## STEP 3. DB Connection Info

```
Host: ec2 public ip
DB name: new_db  
Username: new_user  
password: new_password
port: 5432
```
