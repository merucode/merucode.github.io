---
layout: default
title: Crawling Note
parent: Crawling
grand_parent: Python
nav_order: 99
---

# Crawling Note
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

## STEP 1. GET DATA
### Step 1-1. Data site
* **KOREA**
	* 서울열린데이터광장 [https://data.seoul.go.kr/](https://data.seoul.go.kr/)
	* 공공데이터포털 [https://www.data.go.kr](https://www.data.go.kr/)
	* e-나라지표 [http://www.index.go.kr/](http://www.index.go.kr/)
	* 국가통계포털 [http://kosis.kr](http://kosis.kr/)
	* 서울특별시 빅데이터 캠퍼스 [https://bigdata.seoul.go.kr/](https://bigdata.seoul.go.kr/)
	* 통계청 [http://kostat.go.kr/](http://kostat.go.kr/)
* **ABROAD**
	* 구글 데이터 검색 [https://toolbox.google.com/datasetsearch](https://toolbox.google.com/datasetsearch)
	* 캐글 [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
	* Awesome Public Datasets Github [https://github.com/awesomedata/awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets)
	* Data and Story Library [https://dasl.datadescription.com/](https://dasl.datadescription.com/)
	* 데이터허브 [https://datahub.io/](https://datahub.io/)

<br>

### Step 1-2. Web
* **Web scrapping** : the data fields you want to extract from specific websites
* **Web crawling** : you want to find the urls

<br>

## STEP 2. WEB CRAWLING

### Step 2
* **site address**
[image_데이터-만들기_2-7. 웹사이트 주소 이해하기 01:07]
[image_데이터-만들기_2-7. 웹사이트 주소 이해하기 02:48]

* **parsing** : analyze into its parts, python use Beautiful Soup

<br>

### Step 2-2. Example
```python
import requests
from bs4 import BeautifulSoup
# headers 지정 
headers = { 'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36', }

response = requests.get('url', headers=headers)

### BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')	# change bs type

### soup.select('css_selector')
li_tags = soup.select(".class1 li") 	# 'class1' 클래스를 가진 태그에 중첩된 모든 <li> 태그 선택
li_tags[0].text	 				# 첫 번째 요소 text 추출  
li_tags[0].text.strip()			# 첫 번째 요소 공백 제거한 text 추출

img_tags = soup.select('img') 	# 모든 <img> 태그 선택하기 
img_tags[0]["src"]				# 첫 번째 요소 src 속성 값 추출
```
<br>

* **빈 페이지 구분하기**  : 빈 페이지에만 있는 특수한 선택자 사용
```python
# 첫 페이지 번호 지정 
page_num = 1  

while  True: 
	response = requests.get("http://www.ssg.com/search.ssg?target=all&query=nintendo&page=" + str(page_num), headers=headers)
	soup = BeautifulSoup(response.text, 'html.parser') 

	###  빈 페이지에만 있는 특수한 class 선택자를 사용
	# 빈 페이지에만 사용되는 ".csrch_tip" 클래스가 없을 때만 HTML 코드를 리스트에 담기  
	# 빈 페이지에 걸리는 경우(else) break
	if  len(soup.select('.csrch_tip')) == 0: 
		pages.append(soup) 
		print(str(page_num) + "번째 페이지 가져오기 완료") 
		page_num += 1 
		time.sleep(3) 
	else: 
		break
```

<br>

* **to Dataframe**

```python
import time 
import pandas as pd 
import requests 
from bs4 import BeautifulSoup 


records = [] 	# 빈 리스트 생성 
page_num = 1 	# 시작 페이지 지정 

while  True: 
	response = requests.get("http://www.ssg.com/search.ssg?target=all&query=nintendo&page=" + str(page_num)) 
	soup = BeautifulSoup(response.text, 'html.parser') 
	
	# "prodName" 클래스가 있을 때(페이지가 존재할 때)만 상품 정보 가져오기  
	if  len(soup.select('.csrch_tip')) == 0: 
		product_names = soup.select('.cunit_info > div.cunit_md.notranslate > div > a > em.tx_ko') 
		product_prices = soup.select('.cunit_info > div.cunit_price.notranslate > div.opt_price > em') 
		product_urls = soup.select('.cunit_prod > div.thmb > a > img') 
		page_num += 1 
		time.sleep(3) 
		
		# 상품의 정보를 하나의 레코드로 만들고, 리스트에 순서대로 추가하기  
		for i in range(len(product_names)):
			record = [] 
			record.append(product_names[i].text) 		
			record.append(product_prices[i].text.strip()) 
			record.append("https://www.ssg.com" + product_urls[i].get('src'))
			records.append(record) 
	else: 
		break  

# DataFrame 만들기 
df = pd.DataFrame(data = records, columns = ["이름", "가격", "이미지 주소"]) 
``` 
