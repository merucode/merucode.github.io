---
layout: default
title: Design
parent: Web
grand_parent: Project
nav_order: 1
---

# Design
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
## STEP 1. STRUCTURE

```mermaid
flowchart LR
    a1[VM2]--DATA-->a2[CLOUDSQL]
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

```mermaid
gantt
    dateFormat  YYYY-MM-DD
    title       Web Chart Service

    section HEAD PLANE
    Complete WEB              :done,    des1, 2023-04-01,2023-04-10
    Active DATA               :active,  des2, 2023-04-11, 3d
    Future task               :         des3, after des2, 3d
    Future task2              :         des4, after des3, 3d

    section DATA
    Completed task in the critical line :crit, done, 2023-04-11,24h
    Implement parser and jison          :crit, done, after des1, 2d
    Create tests for parser             :crit, active, 3d
    Future task in critical line        :crit, 5d
    Create tests for renderer           :2d
    Add to mermaid                      :1d
    
    section BACKEND
    Describe gantt syntax               :active, a1, after des1, 3d
    Add gantt diagram to demo page      :after a1  , 20h
    Add another diagram to demo page    :doc1, after a1  , 48h

    section FRONTEND
    Describe gantt syntax               :after doc1, 3d
    Add gantt diagram to demo page      :20h
    Add another diagram to demo page    :48h
```

- [x] WEB_Connect django, guricorn, nginx in VM instance
- [ ] DATA_Get data instance and connect it and cloudsql
- [ ] BACKEND_Connect cloudsql and django
- [ ] FRONTEND_Make front page including data chart