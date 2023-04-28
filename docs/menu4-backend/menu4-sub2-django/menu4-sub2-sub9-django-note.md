---
layout: default
title: Backend
parent: Django
grand_parent: Django Note
nav_order: 9
---

# Django Note
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

## STEP 1. Create Superuser in Ubuntu

### Step 1-1. Error

* `bash`

  ```bash
  $ docker exec b82a python manage.py createsuperuser
  # Superuser creation skipped due to not running in a TTY. You can run `manage.py createsuperuser` in your project to create one manually.
  ```

### Step 1-2. Solution

* `bash`(use -it option)

  ```
  docker exec -it [container-id] python manage.py createsuperuser
  ```

  

<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 2. Connect between CloudSQL to Django

* [Connecting Django to a PostgreSQL Database in Cloud SQL ](https://dragonprogrammer.com/connect-django-database-cloud-sql/)

### Step 2-1. Creating a PostgreSQL Instance in Cloud SQL

1. Go to the [Cloud SQL Instances page](https://console.cloud.google.com/sql/instances).
2. Click *Create instance*.
3. Select *PostgreSQL* and click *Next*.
4. Enter a name. Do not include sensitive or personally identifiable information in your instance name as it is externally visible.
5. Enter a password for the `postgres` user.
6. Configure instance settings under *Configuration options*.

For *Connectivity*, select *Public IP* unless you want to do a more complex setup.

See the documentation notes linked above for details.

7. Click *Create*.

### Step 2-2. Creating a Database and User for Django

1. In the [GCP Console](https://console.cloud.google.com/), open Cloud Shell.

2. Use the built-in `psql` client to connect to your Cloud SQL instance:

   ```bash
   $ gcloud sql connect [INSTANCE_ID] --user=postgres
   ```

3. Enter the password for the `postgres` user you’ve set in the previous section.

4. Now, from the `psql` command-line, create a database for Django:

   ```bash
   $ CREATE DATABASE my_django_db;

5. Create a Django user and password:

   ```bash
   $ CREATE USER my_django_user WITH PASSWORD 'password_for_my_django_user';
   ```

   Of course, choose a good password or use [this list](https://en.wikipedia.org/wiki/List_of_the_most_common_passwords) for suggestions.

6. Set the user’s client encoding to UTF-8:

   ```bash
   $ ALTER ROLE my_user SET client_encoding TO 'utf8';
   ```

7. Grant the Django user rights over the Django database:

   ```bash
   $ GRANT ALL PRIVILEGES ON DATABASE my_django_db TO my_django_user;
   ```

   Replace names as appropriate.

### Step 2-3. Adding the Database Details to `settings.py`

* `.env`(used by `docker-compose.yml`)

  ```dockerfile
  SQL_ENGINE=django.db.backends.postgresql
  SQL_DATABASE=my_django_db
  SQL_USER=my_django_user
  SQL_PASSWORD=[pwsword]
  SQL_HOST=[host ip]
  SQL_PORT=5432
  DATABASE=postgres
  ```

* `settengs.py`

  ```python
  # Before psycopg2-binary should be installed 
  
  import os
  ...
  DATABASES = {
      'default': {
          "ENGINE": os.environ.get("SQL_ENGINE", "django.db.backends.sqlite3"),
          "NAME": os.environ.get("SQL_DATABASE", os.path.join(BASE_DIR, "db.sqlite3")),
          "USER": os.environ.get("SQL_USER", "user"),
          "PASSWORD": os.environ.get("SQL_PASSWORD", "password"),
          "HOST": os.environ.get("SQL_HOST", "localhost"),
          "PORT": os.environ.get("SQL_PORT", "5432"),
      }
  }
  ```

  

### Step 2-4. Make Model Connect with CloudSQL table

* `app/models.py`

  ```python
  from django.db import models
  
  class test_model(models.Model):
      date = models.DateField(primary_key=True, db_column='date')
      price = models.IntegerField(db_column='price')
      
      class Meta:
          managed = False
          db_table = "test_table"	# SQL datatable name
  ```

  * test_table is consist of only 'data' and 'price' columns



### Step 2-5. CloudSQL network 추가

* [SQL] → [연결] → [네트워킹] → 승인된 네트워크 추가



### [TIP] `manage.py inspectdb`

* [Django를 레거시 데이터베이스와 통합하는 방법 ](https://docs.djangoproject.com/en/4.2/howto/legacy-databases/)

* you use command `manage.py inspectdb` , it create `models.py` in django project folder

* The `moels.py` is consist of model code mapping with CloudSQL. So, you use this

  model code as you want 

```bash
$ docker exec python manage.py inspectdb > models.py
### in models.py, you choose code what you want. And copy or paste your app model also possible.
### After that delete models.py create by inspectdb

$ docker exec python manage.py makemigrations
$ docker exec python manage.py migrate
```



