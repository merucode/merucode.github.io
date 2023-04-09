---
layout: default
title: SQL
parent: Database
nav_order: 1
---

# SQL
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
## STEP 0. About SQL
* SQL(Structured Query Language) :  database standard language
* SQL Structure
[image](https://opentutorials.org/course/3161/19535)
  
<br>


<!------------------------------------ STEP ------------------------------------>
## STEP 1. DATABASE(SCHEMA)
```sql
-- Show all database
SHOW DATABASES;					
```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2. TABLE
```sql
-- Show tables
SHOW TABLES; 

-- Show table columns
DESC 테이블명;
```
<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. DATA
```sql
-- Show all data from table
SELECT * FROM 테이블명;

-- Alias(Oavailable for RDER BY, not available for WHERE)  
SELECT * FROM weather w, cities c WHERE w.city = c.name;  -- table alias
SELECT CustomerName AS cu, Address AS ad FROM Customers;  -- column alias

-- Show specific column or row in table
SELECT 열1, 열2 FROM 테이블명 WHERE 조건
SELECT * FROM Customers WHERE Country = 'Germany';
SELECT * FROM Orders WHERE ShipperID !=  2; 
SELECT * FROM OrderDetails WHERE Quantity >  100; 
SELECT * FROM Employees WHERE FirstName >=  'O'; 
SELECT * FROM Employees WHERE BirthDate <=  '1950-01-01'; 

-- Show specific data 
SELECT * FROM 테이블명 WHERE text LIKE '%우아한%'; 	
# %: The number of characters is not specified
# _: The number of characters is specified
SELECT * FROM 테이블명 WHERE number BETWEEN 1 and 3;
SELECT * FROM 테이블명 WHERE text IN (1, 2); 	# same as text = 1 or text = 2
SELECT * FROM 테이블명 WHERE text IS NULL;

-- Sort(내림차순 DESC)  
SELECT 열1, 열2 ... FROM 테이블명 WHERE 조건 ORDER BY 열1 ASC LIMIT 5;
```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 4. JOIN
[image](https://www.bearpooh.com/144)

```sql



```

---
* [](https://365kim.tistory.com/102)