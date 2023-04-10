---
layout: default
title: datetime
parent: Basic
grand_parent: Python
nav_order: 3
---

# datetime
{: .no_toc }



### Step. Repeat day by day

```python
from datetime import datetime, timedelta

# Setting start, last day
start = "2021-08-01"
last = "2021-08-04"

# Convert datetime
start_date = datetime.strptime(start, "%Y-%m-%d")
last_date = datetime.strptime(last, "%Y-%m-%d")

# Repeat untill last date
while start_date <= last_date:
    dates = start_date.strftime("%Y-%m-%d")
    print(dates)

    # Add one day
    start_date += timedelta(days=1)
```