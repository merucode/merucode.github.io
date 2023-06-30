---
layout: default
title: Time
parent: Practice
grand_parent: Python
nav_order: 5
---

# Time
{: .no_toc }



## Step 1. 한국 시간 반영

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