---
layout: default
title: NPL
parent: Web
grand_parent: Project
nav_order: 2
---


* Process

    1. Get data and fill into dataframe
    2. Extract noun and Frequencies per day
    3. NPL data save to CloudSQL

<br>

## STEP 1. Connect

* `connect_tcp.py` (reference: [gcp github])

```python
import os

import sqlalchemy
import pg8000

def connect_tcp_socket() -> sqlalchemy.engine.base.Engine:
    db_host = os.environ["INSTANCE_HOST"]  # Read ENV file in Docker compose
    db_user = os.environ["DB_USER"]  
    db_pass = os.environ["DB_PASS"]
    db_name = os.environ["DB_NAME"] 
    db_port = os.environ["DB_PORT"]

    connect_args = {}
    pool = sqlalchemy.create_engine(
        sqlalchemy.engine.url.URL.create(
            drivername="postgresql+pg8000",
            username=db_user,
            password=db_pass,
            host=db_host,
            port=db_port,
            database=db_name,
        ),
    )
    return pool

### Usage in another file
# from connect_tcp import connect_tcp_socket
# engin = connect_tcp_socket()
# conn = engin.connect()
```

<br>

## STEP 2. Get Data

```python
import 

def stock_post():
    return

def collect_nouns():
    return

def stock_posts():
    return


```


<br>

## STEP 3. main.py

```python
from connect_tcp import connect_tcp_socket

engin = connect_tcp_socket()

##
## 
## to_sql


```

---
[gcp github]: https://github.com/GoogleCloudPlatform/python-docs-samples/tree/72deeb8cfae88229b4710d24730f156f858923f9/cloud-sql/postgres/sqlalchemy