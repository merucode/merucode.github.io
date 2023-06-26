---
layout: default
title: Design
parent: Web
grand_parent: Project
nav_order: 1
---

# Design
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

## STEP 1. STRUCTURE

```mermaid
flowchart LR
    a1[VM2]--DATA-->a2[AWS DB]
    a2--DATA-->b1
    subgraph VM1_Docker
    b1[django]--guricorn-->b2[nginx]
    b3[static]-->b2
    end
    c1[client]--request-->b2
    b2--response-->c1
```

* VM1 : Web instance
* VM2 : Get data instance
* data : word, ohlc

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. To do list

* backend 데이터 전송 확인(확인 완료)
* frontend 에서는 공동 단어를 모름으로 데이터 전송 구조 변형 필요(완료)

* crawling JSON 파일 깨짐(완료)
