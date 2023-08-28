---
layout: default
title: Docker-compose Summary
parent: Docker-compose
grand_parent: Docker
nav_order: 1
---

# Docker-compose Summary
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

```bash
### docker-compose up 과  docker-compose up --build 차이
$ docker-compose up         # 이미지가 없을 때만 빌드하고 컨테이너 시작
$ docker-compose up --build # 이미지 유무에 상관없이 빌드 후 컨테이너 시작

```