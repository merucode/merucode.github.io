---
layout: default
title: Database
parent: Practice
grand_parent: Python
nav_order: 2
---

# Database
{: .no_toc }



## Step 1. Database 최신 날짜 얻기

```python
def get_start_date_from_db(table_name, ticker):
    try:
        with engine.connect() as conn:
            df_check_data = pd.read_sql(f"SELECT * FROM {table_name} WHERE code='{ticker}' ORDER BY date DESC LIMIT 1", con=conn)
            latest_date = df_check_data['date'].values[0]
            start_date = datetime.datetime.strptime(latest_date, '%Y-%m-%d') + datetime.timedelta(days=1)
            start_date = start_date.strftime('%Y%m%d')
            return start_date
    except:
        return '20230620'
```