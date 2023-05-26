---
layout: default
title: Django Model
parent: Django
grand_parent: Backend
nav_order: 8
---

# Django Model
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

## STEP 1. Django Model

### Step 1-1. What is django model

* Django Model : Define the data structure and utilzes the data
* Our Lesson
	1. 데이터베이스 기본 개념 + 장고 모델 연관성
	2. 장고 **migration**
	3. 데이터 간의 관계를 모델로 **표현**
	4. 모델을 통해 데이터 간의 관계를 **활용**


### Step 1-2. Database

* **Database**
	* Relational Database :
		* Save data as shape of table(we use it)
		* SQL
	* Non-Relational Database 
		* Don't save data as shape of table
		* JSON, MongoDB
* **DBMS(Databas Mangement System)**
	* SQlite3, MySQL, PostgreSQL
* **SQL(Structures Query Language)**
	* The language which is communicate with database 

### Step 1-3. Database Table

* **`id` feature**
	* Django **automatically create `id`** field
	* Database **automatically put in `id` as a unique value**
	* `id` field is used for **load data or filter data**
* **Primary Key(`pk`)**
	* Column to **identify** the row
	* Almost case, **primary key = `id`**
* **Use `id` or `pk` whichever is more familiar**

### Step 1-4. Relationship between database and table

* **Foreing key**
	* Use for relation between tables
	* Referencing the **primary key of another table**
* **Type of relationship**(foreing key + 제약조건)
	1. 1:1(user profile)
	2. 1:N (user reviews)
	3. M:N(recommend post)
* django가 알아서 필요한 Foreign key와 제약 조건을 정의

### Step 1-5. ORM

* ORM(Object-Relational Mapper)
	* Django Model → ORM → Database
	* Don't need to wirte SQL by use ORM

<br>
