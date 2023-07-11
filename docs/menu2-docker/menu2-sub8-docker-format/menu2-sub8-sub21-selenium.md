---
layout: default
title: Selenium
parent: Docker Format
grand_parent: Docker
nav_order: 21
---

# Selenium(with docker)
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

## STEP 0. Reference Site
 
* [[Selenium] Ubuntu 20.04에 Chrome driver 설치하기](https://velog.io/@choi-yh/Ubuntu-20.04-selinium-%EC%84%B8%ED%8C%85)
* [DevToolsActivePort file doesn't exist error 해결법](https://study-grow.tistory.com/entry/DevToolsActivePort-file-doesnt-exist-error-%ED%95%B4%EA%B2%B0%EB%B2%95)

<br>

## STEP 1. Docker Code

### Step 1-1. File Structure

* **File structure**

  ```bash
  .
  ├── 📁app
  │   ├── 📄chromdriver
  │   ├── 📄Dockerfile
  │   ├── 📄test.py
  │   └── 📁chrome
  └── 📄docker-compose.yml
  ```

### Step 1-2. Docker Code

* `docker-compose.yml`

```dockerfile
version: '3.7'

services:
  selenium:
    container_name: selenium
    build:
      context: ./app/
      dockerfile: Dockerfile
    volumes:
      - ${PWD}/app/:/usr/src/app
    ports:
      - 8081:8081
    command: /bin/sh # run -it mode
    stdin_open: true # run -it mode  
    tty: true        # run -it mode
```

* `app/Dockerfile`

```dockerfile
FROM python:3.10

WORKDIR /usr/src/app

# Install pakages
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt
```

* `app/requirements.txt`

```bash
selenium
```

### Step 1-3. Chrome 및 Chrome Driver 설치

* `bash`

```bash
$ docker compose up -d --build
$ docker exec -it selenium /bin/bash

> mkdir chrome
> cd chrome
# 설치파일 받기
> wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# 크롬 설치
> apt install ./google-chrome-stable_current_amd64.deb 
# 크롬 버전 확인
> google-chrome --version


```

* https://chromedriver.chromium.org/downloads 에서 chrome과 같은 버전 링크 접속
  * chromedriver_linux64.zip 우클릭 후 링크 주소 복사

* `bash`

```bash
# 복사한 chromedriver 링크로 부터 chromedriver 다운로드
> wget https://chromedriver.storage.googleapis.com/89.0.4389.23/chromedriver_linux64.zip
# 압축 해제 
> unzip chromedriver_linux64.zip
> mv chromedriver ../
```

### Step 1-4. test 코드 작성 및 test

* `test.py``

```python
# selenium의 webdriver를 사용하기 위한 import
from selenium import webdriver

# selenium으로 키를 조작하기 위한 import
from selenium.webdriver.common.keys import Keys

# 페이지 로딩을 기다리는데에 사용할 time 모듈 import
import time

# DevToolsActivePort file doesn't exist error 해결을 위한 options 추가
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument("--single-process")
chrome_options.add_argument("--disable-dev-shm-usage")

# selenium 4.6.0 버전 이상 부터는 경로 없어도 chromedriver 알아서 찾아줌
driver = webdriver.Chrome(options=chrome_options) 

#크롬 드라이버에 url 주소 넣고 실행
driver.get('https://www.google.co.kr/')

# 페이지가 완전히 로딩되도록 3초동안 기다림
time.sleep(3)

# 현재 페이지 url 출력
print(driver.current_url)
driver.close
```

<br>
