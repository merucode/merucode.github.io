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
    * 레코드 유형 : 레코드(트래픽을 IPv4 주소에 라우팅)
    * 레코드 이름 : `@.생성한 domain 주소`
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
          server_name 생성한_domain_주소;	# Apply domain address
  
          ...
          }
  }
  ```

  

<br>



## STEP 2. SSL





```
certbot --nginx -d yourdomain.com
```

까지 수행 



docker에 어떻게 복사할지 키파일
