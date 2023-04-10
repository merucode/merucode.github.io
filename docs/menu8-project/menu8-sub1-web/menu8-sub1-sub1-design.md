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
    Active DATA               :active,  des2, 2023-04-10, 12d
    Future task               :         des3, after des2, 7d
    Future task2              :         des4, after des3, 7d

    section DATA
    Connect CloudSQL                    :crit, done,  a1, after des1, 1d
    Get Data and NPL Processing         :crit, active,a2, after a1, 2d
    Check to Connect and Update         :crit,        a3, after a2, 3d
    Get All Data                        :crit,        a4, after a3, 3d
    Automate data collection(Server)    :crit,        a5, after a4, 3d
    
    section BACKEND
    Connect Django and CloudSQL         :b1, after a5, 3d
    Add gantt diagram to demo page      :b2, after b1, 2d
    Add another diagram to demo page    :b3, after b2, 2d

    section FRONTEND
    Describe gantt syntax               :c1, after b3, 3d
    Add gantt diagram to demo page      :c2, after c1, 2d
    Add another diagram to demo page    :c3, after c2, 2d
```

- [x] WEB_Connect django, guricorn, nginx in VM instance
- [ ] DATA_Get data instance and connect it and cloudsql
- [ ] BACKEND_Connect cloudsql and django
- [ ] FRONTEND_Make front page including data chart