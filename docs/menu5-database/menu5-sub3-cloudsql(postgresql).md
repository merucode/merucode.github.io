---
layout: default
title: cloudSQL
parent: Database
nav_order: 3
---

# CloudSQL(Postgresql)
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

## STEP 1. CloudSQL

### Step 1-1. CloudSQL console

![image-20230419110653118](./../../images/menu5-sub3-cloudsql(postgresql)/image-20230419110653118.png)

### Step 1-2. CloudSQL Connect

* `bash`

  ```bash
  $ gcloud sql connect stock-post --user=[user_id]
  # gcloud not using root user
  ```

* **Example**

  ```bash
  $ gcloud sql connect stock-post --user=postgres
  ```



<br>

<!------------------------------------ STEP ------------------------------------>

### STEP 2. Basic CMD


---

```
# To show a list of databases:
postgres=# \l

# To connect to a specific database:
postgres=# \c DATABASE_NAME

# To show a list of tables:
DATABASE_NAME=# \dt

# To create a table:
DATABASE_NAME=# CREATE TABLE table_name (
    column1 datatype1,
    column2 datatype2,
    column3 datatype3
);

# To insert data into a table:
DATABASE_NAME=# INSERT INTO table_name (column1, column2, column3)
    VALUES (value1, value2, value3);

# To select data from a table:
DATABASE_NAME=# SELECT * FROM table_name;

# To drop a table:
DATABASE_NAME=# DROP TABLE table_name;
```

