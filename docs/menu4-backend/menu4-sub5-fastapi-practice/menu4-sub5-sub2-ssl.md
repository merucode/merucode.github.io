---
layout: default
title: SSL
parent: Fastapi Practice
grand_parent: Backend
nav_order: 2
---

#  SSL
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

* [Github]

  * [EC2] - [네트워크] - [IPv4 방화벽 규칙] : port 80, 443, 5432 추가

  * ENV 설정 : `.backend.env`, `frontend/.env`, `.database.env` 

  * `nginx.conf` domain 적용

  * nginx SSL 발급

    ```bash
    $ docker compose up -d --build
    $ docker exec -it nginx /bin/sh
    > certbot certonly --nginx -d [도메인 주소].co.kr -d www.[도메인 주소].co.kr
    ```

  * `nginx.conf`  SSL 발급 후 주석 처리 해제

    ```nginx
    server {
    	...
        listen 443 ssl; # 주석 처리 해제
        server_name [domain_주소] www.[domain_주소];  
      
        ssl_certificate /etc/letsencrypt/live/[domain_주소]/fullchain.pem;  # 주석 처리 해제
        ssl_certificate_key /etc/letsencrypt/live/[domain_주소]/privkey.pem;# 주석 처리 해제
        include /etc/letsencrypt/options-ssl-nginx.conf; # 주석 처리 해제
        ...
    ```

  * [Create test table](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub11-fastapi-react-postgresql-nginx.html#step-4-connect-with-database)



## STEP 1. Code

### Step 1-1. nginx ssl 인증서 발급

* [[Backend] - [Nginx] - [Domain SSL]](https://merucode.github.io/docs/menu4-backend/menu4-sub8-nginx/menu6-sub6-sub11-domain-ssl.html)

  

### Step 1-2. Code

* `frontend/.env`

  ```bash
  REACT_APP_BACKEND_URL=https://[domain]/api/ 	#https://test.co.kr/api/	# Apply domain
  ```

* 





### Step 1-3. 관련 Error 이슈 해결

* [Websocket Connection Fail](https://merucode.github.io/docs/menu3-frontend/menu3-sub6-react/menu6-sub6-sub99-react-trouble-shooting.html#step-1-websocket-connection-fail)

* [`Mixed Content Error`](https://merucode.github.io/docs/menu3-frontend/menu3-sub6-react/menu6-sub6-sub99-react-trouble-shooting.html#step-1-mixed-content)

* [Backend 요청시 `net::ERR_CERT_COMMON_NAME_INVALID`]()

