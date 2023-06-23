---
layout: default
title: pandas Advanced
parent: pandas
grand_parent: Python
nav_order: 3
---

# pandas Advenced
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

### `df.apply`

* [https://kibua20.tistory.com/194](https://kibua20.tistory.com/194)





## STEP 1. to_sql에서 데이터 형식 지정하기



### Step 1-9. dtype JSON 형식 한글 깨짐 문제 해결

* 데이터베이스 연결 시 인코딩 옵션 설정

  ```python
  ...
  	connect_args = {}
      pool = sqlalchemy.create_engine(
          sqlalchemy.engine.url.URL.create(
              drivername="postgresql+pg8000",
              username=db_user,
              password=db_pass,
              host=db_host,
              port=db_port,
              database=db_name,
              query={'client_encoding': 'utf8'}  # JSON 파일 한글 깨짐 관련 인코딩 설정
          ),
      )
      return pool
  ...
  ```

  
