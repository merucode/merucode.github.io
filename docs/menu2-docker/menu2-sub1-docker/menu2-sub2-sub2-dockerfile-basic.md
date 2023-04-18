---
layout: default
title: Docekerfile Basic
parent: Dockerfile
grand_parent: Docker
nav_order: 2
---

# Dockerfile Basic
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



## STEP . Run

* —no-cache-dir
  * 캐시를 사용하지 않고 pip install을 하라는 뜻. 하드디스크에 공간이 없거나 도커 이미지를 작게 유지하고 싶을 때 사용한다.		

```docker
# install package
COPY requirements.txt requirements.txt
RUN pip install --nocache-dir -r requirements.txt
```
