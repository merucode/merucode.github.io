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

## STEP 1. Show Model Data

* `bash`

  ```bash
  python manage.py shell
  >> from ap  p_name.models import User, Review    # app의 User, Review 모델 import
  >> User.objects.all()                     # User 모델 객체 모든보기
  >> for user in User.objects.all():        # User 모델 칼럼 데이터 확인
        print(user.email, user.email_domain)  
  ```

## STEP 2. Show SQLite3 Data

### Step 2-1. VSCode "SQLite Viewer" Extension

### Step 2-2. Djagno Database Shell

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


