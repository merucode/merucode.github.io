---
layout: default
title: Design
parent: Web
grand_parent: Project
nav_order: 1
---

# menu6 sub1 sub1
{: .no_toc }


```mermaid
flowchart TB
    subgraph VM1 
    a1[VM1]--data-->a2[CloudSQL]
    end
    a2--"data"-->b1
    subgraph VM_Docker
    b1[django]--guricorn-->b2[nginx]
    end
```

* data : word, ohlc