---
layout: default
title: Docekerfile 2nd Build
parent: Dockerfile
grand_parent: Docker
nav_order: 8
---

# Dockerfile 2nd Build
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

## STEP 1. Example

* `Dockerfile.prod`

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

* `entrypoint.prod.sh`

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