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
```

<br>

## STEP 2. Get Data

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

### NPL library
from konlpy.tag import Okt
from nltk import Text
from collections import Counter

### Ready for konlpy
mecab = Mecab()

### Ready for crawling
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}  

### Crawling post
def post_to_df(code, engine):
    url = f"https://finance.naver.com/item/board.naver?code={stock_code}&page="

    res = requests.get(url+'1', headers=headers)
    soup = BeautifulSoup(res.text, "lxml")

    page_last = soup.find("td", attrs={"class":"pgRR"}).find("a")["href"].split('=')[-1]

    df_posts = pd.DataFrame()

    for i in range(1, 15):#int(page_last)+1
        url_page = url + str(i)
        res = requests.get(url_page, headers=headers)

        # Get post date, title
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', {'class': 'type2'})
        df = pd.read_html(str(table))[0]
        df = df[['날짜','제목','조회']]
        df.dropna(inplace=True)

        # Get post content
        post_urls = ['https://finance.naver.com' + link['href'] for link in table.select('.title > a')]
        post_contents = []
        for post_url in post_urls:
            post_html = requests.get(post_url)
            post_soup = BeautifulSoup(post_html.content, 'html.parser')
            post_content = post_soup.find('div', {'class': 'view_se'}).get_text().strip()
            post_contents.append(post_content)
        df['content'] = post_contents

        # Get the most recent date in the SQL data for this stock code
        # 업데이트 관련 날짜 처리 추가 고민 필요
        df['날짜'] = pd.to_datetime(df['날짜'])
        try:
            recent_date = pd.read_sql(f"SELECT date FROM {code} ORDER BY date DESC LIMIT 1", conn)['date'][0]
        except:
            recent_date = pd.to_datetime('1990-01-01')


        # Filter only the more recent data
        df = df[df['날짜'] > recent_date]

        # 이전 데이터 날짜 부터 for문 끝내기    
        if df.empty:
            break
        
        df_post = pd.concat([dfs,df], ignore_index=True)
    
    df['날짜'] = df["날짜"].dt.strftime("%Y-%m-%d") # Change date format
    df_posts.rename(columns={'날짜':'date', '제목':'title'}, inplace=True)
    return df_posts

### NPL
def col_npl(df_posts, engine, code):

    ### Loop by day
    end_date = df.iloc[0, 1]
    start_date = df.iloc[-1, 1]

    # 시작일, 종료일 datetime 으로 변환
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")


    # 종료일 까지 반복
    df_cols = pd.DataFrame(columns=['date', '1st','1st_count','2nd','2nd_count','3rd','3rd_count','4th','4th_count','5th','5th_count'])
    while start_date <= last_date:
        date = start_date.strftime("%Y-%m-%d")

        df_post = df_posts.loc[[df_post['date']==date]]
        titles = df_post['title'].tolist()
        contents = df_post['content'].tolist()


        ##### ? df.apply 가능 
        ##### ? tokens = sum(df['cleaned_tokens'], [])
        ##### ? 위와 비슷하게 사용하면 열의 문자열을 리스트 반환??
        
        for title, content in zip(titles, contents):
            pharse = title + content
            nonus = mecap.nouns(pharse)
            for i, j in enumerate(nouns):
                if len(j) < 2:  # Remove one letter noun
                    noun.pop(i)
            count = Counter(nouns)

            noun_list = count.most_common(10)
            for i in noun_list:
                print(i)   
            ### to df_col 처리 필요        


        df_cols = pd.concat([df_cols,df_col])			# 열병합이 default

        # 하루 더하기
        start_date += timedelta(days=1)

    # df_cols.to_sql(f'{code}, con=engine, if_exists='append', index=False)
    pass

### 
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