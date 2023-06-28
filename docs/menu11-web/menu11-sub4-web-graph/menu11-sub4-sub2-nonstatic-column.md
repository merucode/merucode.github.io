---
layout: default
title: Non Static Columns Data Graph
parent: Web Graph
grand_parent: Web
nav_order: 2
---

# Non Static Columns Data Graph

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

## STEP 0. Reference Site

* [Github]()

## STEP 1. Create Test Data

### Step 1-0. About Data

* Non Static Columns → Handling using JSON type database
* Example Data

	|date|words_count|code|
	|---|---|---|
	|2023-06-01|{"col1": 3, "col2": 5, "col3": 20}|000001|
	|2023-06-02|{"col1": 5, "col2": 7}|000001|
	|2023-06-12|{"col2", 5, "col3": 10}|000001|
	|2023-06-20|{"col4": 13}|000001|
	|2023-06-07|{"col1": 15, "col3": 25}|000002|
	|2023-06-08|{"col6": 7, "col1": 5}|000002|
	|2023-06-12|{"col7": 5}|000002|



### Step 1-1. Setting
* Start From [fastapi-react-postgresql-nginx basic form](https://github.com/merucode/form/tree/fastapi-react-postgresql-nginx_basic)
* Setting ENV files : `.backend.env`, `.database.env`, `frontend/.env`


### Step 1-2. Create Test Table

* `bash`

	```bash
	$ docker compose up -d --build
	$ docker exec -it database /bin/bash
	> su - postgres
	> psql -U test_user -d test_db;
	> DROP TABLE test_table;
	> CREATE TABLE test_table (
		date DATE NOT NULL, 
		words_count JSON NOT NULL,
	    code VARCHAR NOT NULL
	);
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-01','{"col1": 3, "col2": 5, "col3": 20}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-02', '{"col1": 5, "col2": 7}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-12','{"col2": 5, "col3": 10}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-20','{"col4": 13}', '000001');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-07','{"col1": 15, "col3": 25}', '000002');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-08','{"col6": 7, "col1": 5}', '000002');
	INSERT INTO test_table (date, words_count, code) VALUES ('2023-06-12','{"col7": 5}', '000002');
	```

<br>

## STEP 2.