---
layout: default
title: Nginx Trouble Shooting
parent: Nginx
grand_parent: Backend
nav_order: 99
---

# Nginx Trouble Shooting
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

## STEP 1. SSL_do_handshake() failed (SSL: error:0A00010B ...)

### Step 1-1. Trouble
* url 접속 시 `# 502 Bad Gateway` 화면
* nginx 로그 확인 시 ` SSL_do_handshake() failed (SSL: error:0A00010B ...)` 에러 발생

### Step 1-2. Cause
* [https://stackoverflow.com/questions/76234540/ssl-do-handshake-failed-ssl-error0a00010bssl-routineswrong-version-numbe](https://stackoverflow.com/questions/76234540/ssl-do-handshake-failed-ssl-error0a00010bssl-routineswrong-version-numbe)

### Step 1-3. Solution

* `nginx/nginx.conf`
	* 변경 전
		```nginx
		location / {
		  proxy_pass https://www.example.com;
		  ...
		}
		```
	* 변경 후
		```nginx
		location / {
		  proxy_pass http://www.example.com;
		  ...
		}
		```

