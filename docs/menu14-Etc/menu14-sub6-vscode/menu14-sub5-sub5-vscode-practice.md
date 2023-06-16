---
layout: default
title: Vscode Practice
parent: Vscode
grand_parent: Etc
nav_order: 2
---

# Vscode Practice

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



## STEP 0. Related Site

* [https://gre-eny.tistory.com/344](https://gre-eny.tistory.com/344)





## STEP 1. Connect EC2 Instance

### Step 1-1. Config 파일 생성

* Extension : `Remote Development` 설치

* [F1] → `Remote-SSH: Open SSH Configuration File ...`

* `사용자이름/.ssh/` 에 SSH key 파일(`~.pem`) 넣기(ec2 ssh 기본키 생성)

* `config`

  ```python
  Host ami2_jenkins_ec2
    HostName [퍼블릭 IPv4 DNS 주소]
    User [ec2-user]
    IdentityFile ~/.ssh/[shh_key.pem]
  ```

  - `Host`: 연결할 ec2의 별칭(별명)이므로 아무렇게나 지어도 상관없다.
  - `HostName`: ec2의 `퍼블릭 IPv4 DNS 주소`이다. 위 사진에서는 `노란색 박스` 부분이다.
  - `User`: 접속할 EC2의 사용자 이름
  - `IdentityFile`: ec2에 접속할 때 필요한 `.pem`키의 위치를 지정한다. 본인은 `config` 파일과 같은 폴더 내에 위치한다.



### Step 1-2. 원격 접속하기

* [F1] → `Remote-SSH: Connect to Host ...`
* Host 별칭으로 접속

