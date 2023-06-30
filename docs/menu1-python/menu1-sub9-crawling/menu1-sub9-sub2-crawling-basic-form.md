---
layout: default
title: Crawling Basic Form
parent: Crawling
grand_parent: Python
nav_order: 2
---

# Crawling Basic Form
{: .no_toc }



## Step 1. 요청 후 일정 시간 경과 후 요청

### Step 1-1. File Structure

* File Structure

  ```bash
  .
  ├── 📁app
  │   ├── Dockerfile
  │   ├── main.py
  │   ├── requirements.txt
  │   └── 📁src
  │       ├── try_request.py
  │       └── util.py
  └── docker-compose.yml
  ```



### Step 1-2. Docker 

* `docker-compose.yml`

    ```Docker
    version: '3.8'
    
    services:
      crawling:
        container_name: crawling
        build:
          context: ./app/
          dockerfile: Dockerfile
        volumes:
          - ${PWD}/app/:/usr/src/app
        env_file:
          - .env
        ports:
          - 8080:8080
        command: python -u main.py
    ```

* `app/Dockerfile`

    ```Docker
    FROM python:3.10
    
    WORKDIR /usr/src/app
    
    ENV JAVA_HOME /usr/lib/jvm/java-1.7-openjdk/jre
    RUN apt-get update && apt-get install -y g++ default-jdk
    
    # install pakages
    RUN pip install --upgrade pip
    COPY ./requirements.txt .
    RUN pip install -r requirements.txt
    ```

* `app/requirements.txt`

    ```bash
    ### Basic
    pandas
    numpy
    asyncio
    
    ### Crawling
    requests
    beautifulsoup4
    lxml
    
    ### DB
    sqlalchemy
    pg8000
    psycopg2-binary
    ```

    

### Step 1-3. Code 

* `app/main.py`
    
    ```python
    import time
    import asyncio
    
    from src.try_request import request_crawling
    from src.util import korea_time
    
    async def task():
        while True:
            now_time = korea_time()
            print(f"Task executed at: {now_time}")
    
            # Here add your task
    		request_crawling()
            
            await asyncio.sleep(1 * 60 * 60)
    
    
    async def main():
        print("Starting the task...")
        await asyncio.gather( 
            task(),  
            )
    """
    # Several task operate example : 
    	await asyncio.gather( 
    		task(), 
    		task() 
    	)
    """
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    ```
    
* `app/src/try_request`

    * 크롤링 중 에러 처리 및 재요청
    * 과도한 크롤링 요청 시 요청 서버에서 막는 경우 처리

    ```python
    import time
    import requests
    
    from src.util import korea_time
    
    def request_crawling(code):
        try:
            # Your code to send a request to the server would go here
            now_time = korea_time()
            print(f"# OPERATE: request crawling \n# Log time: {now_time}")
            time.sleep(10)
            print('# COMPLETE: complete request crawling')
    
        except requests.exceptions.RequestException as e:
            # Handle any request exceptions, such as a network error or server issue
            now_time = korea_time()
            print(f"# ERROR: Error occurred: {e} \n# Log time: {now_time}")
            time.sleep(3600) # Wait 3600 seconds and try again
    
        except Exception as e:
            # Handle any unexpected exceptions
            now_time = korea_time()
            print(f"# ERROR: Unexpected error occurred: {e} \n# Log time: {now_time}")
    
        pass   
    ```

* `app/src/util.py`

  ```python
  import datetime
  import pytz
  
  def korea_time():
      # Get the current time in UTC
      utc_now = datetime.datetime.now(pytz.utc)
  
      # Convert the UTC time to the Korean time zone
      korean_tz = pytz.timezone('Asia/Seoul')
      korean_time = utc_now.astimezone(korean_tz)
  
      # Format the Korean time
      formatted_time = korean_time.strftime('%Y-%m-%d %H:%M:%S')
  
      return formatted_time
  ```

  