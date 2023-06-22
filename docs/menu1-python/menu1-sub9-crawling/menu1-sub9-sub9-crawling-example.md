---
layout: default
title: Crawling Example
parent: Crawling
grand_parent: Python
nav_order: 1
---

# Crawling Example
{: .no_toc }



## Step 1. Naver 종목토론실 게시글 Crawling

### Step 1-1. File Structure

    ```
    .
    ├── 📁app
    │   ├── Dockerfile
    │   ├── main.py
    │   ├── requirements.txt
    │   └── 📁src
    │       ├── connect_tcp.py
    │       ├── naver_stock_post.py
    │       ├── nlp.py
    │       └── try_request.py
    ├── docker-compose.prod.yml
    └── docker-compose.yml
    ```

### Step 1-2. Docker 

* `docker-compose.yml`

    ```Docker
    version: '3.7'

    services:
    app:
        container_name: myapp
        build:
        context: ./app/
        dockerfile: Dockerfile.dev
        volumes:
        - ${PWD}/app/:/usr/src/app
        env_file:
        - ./.env  # For DB
        ports:
        - 80:80
        command: /bin/bash   # To be possible : docker exec -it myapp /bin/bash
        stdin_open: true     # ..
        tty: true            # ..
    ```

* `docker-compose.prod.yml`

    ```Docker
    version: '3.7'

    services:
    app:
        container_name: myapp
        build:
        context: ./app/
        dockerfile: Dockerfile.dev
        volumes:
        - ${PWD}/app/:/usr/src/app
        env_file:
        - ./.env
        ports:
        - 8080:80
        command: python -u main.py  # -u option : Show python logs
    ```

* `app/Dockerfile`

    ```Docker
    FROM python:3.9

    WORKDIR /usr/src/app

    RUN pip install --upgrade pip
    COPY ./requirements.txt .
    RUN pip install -r requirements.txt
    ```

### Step 1-3. Code 

* `app/main.py`
    * 일정 시간마다 크롤링 요청

    ```python
    import time
    from src.try_request import request_crawling

    while True:
        request_crawling()
        time.sleep(36000) # Wait 10 hour before sending the next request
    ```

* `app/src/try_request`
    * 크롤링 중 에러 처리 및 재요청
    * 과도한 크롤링 요청 시 요청 서버에서 막는 경우 처리

    ```python
    import time
    import requests
    from src.post import  get_word_counts_from_post

    codes = [...]

    def request_crawling():
        try:
            for code in codes:
                get_word_counts_from_post(code) 
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            time.sleep(3600) # Wait 1 hour and try again
            get_word_counts_from_post(code)
        except Exception as e:
            print(f"unexpected error occurred: {e}")
            time.sleep(3600) # Wait 1 hour and try again
            get_word_counts_from_post(code)
        pass
    ```

* `app/src/post.py`
    * 크롤링 코드

    ```python
    import pandas as pd
    import time
    import datetime
    from datetime import timedelta
    from collections import Counter

    from bs4 import BeautifulSoup
    import requests

    from src.connect_tcp import connect_tcp_socket
    from src.nlp import collect_nouns

    ### Ready for crawling
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "referer":"https://finance.naver.com/item/board.naver?"  
        }
    # 100 page 이상 부터 Referrer-Policy : unsafe-url 관련 referer 추가
    engine = connect_tcp_socket() # DB connect engine 생성
    stock_name_dict = {...}

    ### Main control code
    def get_word_counts_from_post(code):
        stock_name = stock_name_dict[f'{code}']
        url = f"https://finance.naver.com/item/board.naver?code={code}&page=" # 크롤링 URL 설정

        ## Get post crawling from website(post date, title, views) as df
        df_posts = get_df_post(code, url)

        ## daily_divid_posts and return daily word_counts
        df_nlp = daily_divid_posts_nlp(df_posts, stock_name)
        df_nlp['code'] = code       # Add stock code
    
        ## Save nlp dataframe
        with engine.connect() as conn:
            df_nlp.to_sql('test', con=conn, if_exists='append',index=True)
        pass


    ### Conver post to dataframe and Handling initalize or update
    def get_df_post(code, url) -> pd.DataFrame:
        latest_date = get_latest_date_update(code) # Get latest data from DB

        if not latest_date: # DB에 데이터가 없다면 Initialize
            page_last = get_last_page_init(url)
            df_posts = post_to_df_init(page_last, url)
        else:               # DB에 데이터가 있다면 Updaet
            df_posts = post_to_df_update(latest_date, url)
        return df_posts


    ### Get latest date from DB
    def get_latest_date_update(code) -> int:
        try:
            with engine.connect() as conn:
                df_check_data = pd.read_sql(f"SELECT * FROM test WHERE code='{code}' ORD
    ER BY date DESC LIMIT 1", con=conn)
            latest_date = df_check_data.iloc[-1, 0] 
            return latest_date
        except:
            return False


    ### Get last page from post
    def get_last_page_init(url) -> int:
        res = requests.get(url+'1', headers=headers)
        soup = BeautifulSoup(res.text, "lxml")
        page_last = int(soup.find("td", attrs={"class":"pgRR"}).find("a")["href"].split(
    '=')[-1])
        return page_last


    ### Convert post to dataFrame as initialize
    def post_to_df_init(page_last, url) -> pd.DataFrame:
        for i in range(page_last, 0, -1):   # 가장 마지막 페이지 → 최근 페이지 순
            url_page = url + str(i)
            res = requests.get(url_page, headers=headers)
            
            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.find('table', {'class': 'type2'})
            df = pd.read_html(str(table))[0]
            df = df[['날짜','제목']]        # Set extract columns
            df.columns = ['date', 'title']
            df['title'] = df['title'].str.split('[').str.get(0)   # Remove number of com
    ments[]
            df.dropna(inplace=True)         # Drop don't need rows
            df = df.iloc[::-1]              # Reserve df
            
            try:    # Merge extract df
                df_posts = pd.concat([df_posts,df], ignore_index=True)
            except: # Prevent code from stopping(first time concat occur error)
                df_posts = df

            time.sleep(10)                          # Time sleep for avoid error(over requests)
        return df_posts


    ### Convert post to dataframe as update
    def post_to_df_update(latest_date, url) -> pd.DataFrame:
        # 기존 latest_date는 DB 상 최근 날짜이므로 중첩 방지를 위해 +1일 추가
        # DB에 6월1일까지 데이터가 있다면, 최근 부터 6월2일 00:00:00 데이터까지 수집
        latest_date = datetime.datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(da
    ys=1) 
        for i in range(1, 1000000):  # 최근 페이지 → 예전 페이지 순
            url_page = url + str(i)
            res = requests.get(url_page, headers=headers)

            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.find('table', {'class': 'type2'})
            df = pd.read_html(str(table))[0]
            df = df[['날짜','제목']]        # Set extract columns
            df.columns = ['date', 'title']
            df['title'] = df['title'].str.split('[').str.get(0)   # Remove number of comments[]
            df.dropna(inplace=True)         # Drop don't need rows

            df['date'] = pd.to_datetime(df['date'])

            df = df[df['date'] > latest_date] # latest date보다 최신 날짜만 반영

            # latest date 보다 최신 데이터가 없다면(빈 df) 더 이상 받을 데이터가 없으므로 return 처리
            if df.empty:       
                df_posts = df_posts.iloc[::-1]              # Reserve df
                return df_posts

            try:    # Merge extract df
                df_posts = pd.concat([df_posts,df], ignore_index=True)
            except: # Prevent code from stopping(first time concat occur error)
                df_posts = df

            time.sleep(10)                 # Time sleep for avoid error(over requests)
        pass

    ### Convert Post dataframe to NLP dataframe 
    def daily_divid_posts_nlp(df_posts, stock_name):
        df_posts['date'] = pd.to_datetime(df_posts['date'])
        df_posts['date'] = df_posts['date'].dt.strftime('%Y-%m-%d') # 비교용 문자열 전환
        now = datetime.datetime.now()
        # 오늘이 6월2일이라면 6월1일 데이터까지 전처리 수행
        end_date = now - timedelta(days=1)
        start_date = datetime.datetime.strptime(df_posts.iloc[0, 0], "%Y-%m-%d")

        col_name = ['date', 'words_count']
        df_collect_words_count = pd.DataFrame(columns=col_name)
        
        while start_date <= end_date:
            date = start_date.strftime("%Y-%m-%d")          # 비교용 문자열 전환
            df_post = df_posts.loc[df_posts['date']==date]  # daily post  
            pharse_list = df_post['title'].tolist()         # + df_post['content'].tolist()

            nouns = collect_nouns(pharse_list, stock_name)  # NPL preprocessing
            count = Counter(nouns)      # 단어 카운터 처리
            if len(count) != 0:
                lst_df_input = [date] + [count.most_common()]   # input data
                df_collect_words_count.loc[len(df_collect_words_count)] = lst_df_input

            # ### Step 6-2. Input value in dataframe
            start_date += timedelta(days=1) 

        return df_collect_words_count
    ```

* `app/src/nlp.py`
    * 문장 → 단어 처리

    ```python
    from konlpy.tag import Okt 
    from nltk import Text 
    from collections import Counter 

    my_stopwords_set = {...} # Set customized stopword(불용어 세트) 
    my_synonym_set = {'김영수':('영수','영수씨'), ...} # Set customized synonym(동의어 세트) 
    
    Okt = Okt() 
    
    def collect_nouns(sen_list, stock_name): 
        nouns=[] 
        for k in sen_list: 
            noun = Okt.nouns(k)             # tokenized
            noun = clean_by_len(noun, 1)    # 단어 길이 
            noun = clean_by_synonym(noun, my_synonym_set) # 동의어
            noun = clean_by_stopwords(noun, my_stopwords_set) # 불용어
            nouns.extend(noun) 
        nouns = clean_by_freq(nouns, 1) # 등장 빈도
        return nouns

    ### 등장 빈도 cleaning  
    def clean_by_freq(tokenized_words, cut_off_count):  
        vocab = Counter(tokenized_words) # print(f'vocab : {vocab}') 
        uncommon_words = {key for key, value in vocab.items() if value <= cut_off_count}
        cleaned_words = [word for word in tokenized_words if word not in uncommon_words]  
        return cleaned_words  
        
    ### 단어 길이 cleaning 
    def clean_by_len(tokenized_words, cut_off_length):  
        cleaned_by_freq_len = []  
        for word in tokenized_words:  
            if len(word) > cut_off_length:  
                cleaned_by_freq_len.append(word)  
        return cleaned_by_freq_len 
    
    ### 불용어 처리    
        def clean_by_stopwords(tokenized_words, stop_words_set):  
        cleaned_words = [] 
        for word in tokenized_words:  
            if word not in stop_words_set:  
                cleaned_words.append(word)  
        return cleaned_words 
    
    ### 동의어 처리 
    def clean_by_synonym(tokenized_words, my_synonym_set): 
        cleaned_words =[] 
        for word in tokenized_words: 
            for key, set_tuple in my_synonym_set.items(): 
                if word in set_tuple: 
                    word = key 
            cleaned_words.append(word) 
        return cleaned_words
    ```

* `app/src/connect_tcp.py`
    * 데이터베이스 연결 pool 생성

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
    ```