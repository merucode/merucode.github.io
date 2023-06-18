---
layout: default
title: Domain SSL
parent: Nginx
grand_parent: Backend
nav_order: 11
---

# Domain SSL
{: .no_toc}

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
<!------------------------------------ STEP ------------------------------------>



## STEP 0. Related Site

* [점프투fastapi](https://wikidocs.net/75563)

* [https://geko.cloud/en/nginx-letsencrypt-certbot-docker-alpine/](https://geko.cloud/en/nginx-letsencrypt-certbot-docker-alpine/)




<br>



##  STEP 1. Domain 발급 및 EC2 연결

### Step 1-1. Domain 발급

* 사용 가능한 domain인지 확인

  * https://whois.kr/

* domain 발급

  * [https://www.gabia.com/](https://www.gabia.com/) 또는

    * 생성 시 네임서버 주소는 일단 해당 업체로 하고 나중에 AWS 네임 서버 추가 예정

  * AWS Route53

    

### Step 1-2. Domain과 EC2 고정 IP 연결

* EC2 Instance 고정 IP 발급된 상태에서 수행
* [Amazon Light Sail] → [도메인 및 DNS] → [DNS 영역 생성] 
  * 다른 도메인 대행업체의 도메인 사용 : 발급 받은 Domian 입력
  * DNS 영역 생성
* 생성된 DNS 영역 
  * [DNS 레코드] → [레코드 추가]
    * 레코드 추가 1
      * 레코드 유형 : 레코드(트래픽을 IPv4 주소에 라우팅)
      * 레코드 이름 : `@.생성한 domain 주소`
      * 확인 : AWS 고정 IP
    * 레코드 추가 2
      * 레코드 유형 : 레코드(트래픽을 IPv4 주소에 라우팅)
      * 레코드 이름 : `www.생성한 domain 주소`
      * 확인 : AWS 고정 IP
  * [도메인] → [이름 서버] 4개
    * gabia에서 domain 발급 받은 경우
      * [my가비아] → [이용중인 서비스 : Domain] → [관리] → [도메인 정보변경] → [네임서버] 
      * gabia 네임서버에 AWS 이름서버 4개 추가



### Step 1-3. Nginx에 Domain 적용

* `nginx/nginx.conf`

  ```nginx
  server {
          listen 80;
          server_name [domain_주소] www.[domain_주소];	# Apply domain address
  
          #...
  }
  ```
  


<br>



## STEP 2. SSL

### Step 2-1. SSL 인증서 발급

* `./docker-compose.yml`

  ```dockerfile
  ...
    nginx:
      container_name: nginx
      build:
        context: ./nginx/
        dockerfile: Dockerfile
      volumes:
        - ${PWD}/nginx/letsencrypt/:/etc/letsencrypt # Add
      ports:
        - 80:80
        - 443:443	# Add
      ...
  ```

* `./nginx/Dockerfile`

  ```dockerfile
  FROM nginx:1.25-alpine
  
  # certbot 의존 파일 설치 # Add
  RUN apk add python3 python3-dev py3-pip build-base libressl-dev musl-dev libffi-dev rust cargo
  RUN pip3 install pip --upgrade
  RUN pip3 install certbot-nginx
  RUN mkdir /etc/letsencrypt
  
  # conf 삭제 후 복사
  RUN rm /etc/nginx/conf.d/default.conf
  COPY nginx.conf /etc/nginx/conf.d/
  ```

* `bash`

  ```bash
  $ docker compose up -d --build
  $ docker exec -it nginx /bin/sh
  > certbot certonly --nginx -d [도메인 주소].co.kr -d www.[도메인 주소].co.kr
  # e-mail 입력 및 y 두번 입력
  # /etc/letsencrypt/live/[domain 주소]/fullchain.pem 및 privkey.pem 생성(vscode에서는 확인안되고 docker exec로 들어가서 cd 및 ls로 확인 가능)
  ```



### Step 2-2. nginx 파일에 적용

* `./nginx/nginx.conf`

  ```nginx
  # 3000번 포트에서 frontend가 돌아가고 있다는 것을 명시
  upstream frontend {
      server frontend:3000;
  }
  
  # 5000번 포트에서 backend서버가 돌아가고 있다는 것을 명시
  upstream backend {
      server backend:8000;
  }
  
  # nginx 서버 80번 → 443으로 리다이렉팅
  server {
          listen 80;
          server_name [domain_주소] www.[domain_주소];
          rewrite        ^ https://$server_name$request_uri? permanent;
  }
  
  server {
      # → nginx ssl 구성 관련 443 포트로 받기
      listen 443 ssl;
      server_name [domain_주소] www.[domain_주소];  # 도메인 추가
  
      ssl_certificate /etc/letsencrypt/live/[domain_주소]/fullchain.pem; # managed by Certbot
      ssl_certificate_key /etc/letsencrypt/live/[domain_주소]/privkey.pem; # managed by Certbot
      include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
  
      # ...
  }
  
  ```

  

### Step 2-3. 갱신 관련(아직 미수행)

* `/bin/letsencrypt_renew.sh`파일 생성

  ```bash
  cd /home/ubuntu/[docker 위치]
  docker compose up -d --build
  docker exec -it nginx /bin/sh
  certbot renew --force-renew
  nginx -s reload
  ```

* crontab 등록

  ```bash
  chmod 750 /bin/letsencrypt_renew.sh
  crontab -e
  
  # 짝수달 3일 0시 0분에 renew 스크립트 실행
  00 00 03 */2 * /bin/letsencrypt_renew.sh
  ```

* [참고] Restart Nginx in Alpine Linux

  ```bash
  $ nginx -s reload
  $ nginx -s [reload | stop | quit | reopen] 
  ```

* 참고 사이트
  * [https://svrforum.com/svr/264459](https://svrforum.com/svr/264459)
  * [https://nirsa.tistory.com/339](https://nirsa.tistory.com/339)
