---
layout: default
title: React Trouble Shooting
parent: React
grand_parent: Frontend
nav_order: 99
---

# React Trouble Shooting
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



## STEP 1. Websocket Connection Fail

### Step 1-1. Trouble

![image-20230618104319429](./../../../images/menu6-sub6-sub99-react-trouble-shooting/image-20230618104319429.png)

### Step 1-2. Cause

* [https://sonnson.tistory.com/44](https://sonnson.tistory.com/44)
* 원인 : cra에서 기본 포트를 3000포트로 설정해서 나는 에러

### Step 1-3. Solution

* **React**

  * .env 파일에서 WDS_SOCKET_PORT=0 설정

* **nginx**

  * /ws 경로에 다음과 같은 설정 추가

    ```nginx
    location /ws {
        proxy_pass http://dev-front-server;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header Origin "";
    }
    ```

    

<br>

