---
layout: default
title: Time
parent: Practice
grand_parent: Python
nav_order: 5
---

# Time
{: .no_toc }


## STEP 1. 단순 시간

### Step 1-1. 한국 시간

```python
import datetime
from pytz import timezone

time = datetime.datetime.now(timezone('Asia/Seoul'))
print(time)
```

<br>

## STEP 2. 함수 관련 

### Step 2-1. Function 실행 및 종료, 실행 시간 확인 데코레이션

```python
import datetime
from pytz import timezone

def time_check(func):                        # 호출할 함수를 매개변수로 받음
    def wrapper():                           # 호출할 함수를 감싸는 함수
        start_time = datetime.datetime.now(timezone('Asia/Seoul'))
        print(func.__name__, f"executed at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")    # __name__으로 함수 이름 출력
        
        func()                               # 매개변수로 받은 함수를 호출

        end_time = datetime.datetime.now(timezone('Asia/Seoul'))
        print(func.__name__, f"finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        diff_time = end_time - start_time
        print(func.__name__, f"taken time : {diff_time}")

    return wrapper                           # wrapper 함수 반환

"""
from util import time_check

@time_check
def func():
    ...
"""  
```