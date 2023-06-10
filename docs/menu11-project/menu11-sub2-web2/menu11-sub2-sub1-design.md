---
layout: default
title: Design
parent: Web2
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
    b1[fastapi]--guricorn-->b2[nginx]
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

* Complete crawling code and collect test data
  * on VM instance
* Connect between fastapi and react
* Add function Oauth2 with google 
