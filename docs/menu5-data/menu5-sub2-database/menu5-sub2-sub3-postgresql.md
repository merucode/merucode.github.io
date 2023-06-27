---
layout: default
title: Postgresql
parent: Database
grand_parent: Data
nav_order: 3
---

# Postgresql
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
## STEP 1. About Postgresql

### Step 1-1. RDBMS

* **security advantage**
    * Management by user authority

* **[Avoid redundant data]**
    * separate table to manage
    * use by joining

<br>


## STEP 2. Postgres CMD

* `bash`

  ```bash
  $ psql                               # postgres DB에 postgres으로 접속
  $ psql -d mydb                       # mydb DB에 postgres으로 접속
  $ psql -U username  -d mydb          # mydb DB에 username으로 접속


  => \du                     # User 목록 보기
  => \l                      # 데이터베이스 목록 보기
  => \d                      # 테이블 목록 보기
  => \d {table_name}   	     # 지정된 테이블의 컬럼 목록 보기
  => \dv                     # 뷰 목록 보기
  => \ds                     # 시퀀스 목록 보기
  => \dn                     # 스키마 목록 보기

  => \c {db_name}            # 다른 DB에 접속
  => \c {db_name} {usr_name} # 다른 DB에 지정된 사용자로 접속

  => \q                       # PSQL 종료(Ctrl+d) 



<br>
