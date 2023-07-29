---
layout: default
title: Data Load
parent: Data
grand_parent: AI Competition
nav_order: 3
---

# Data Load 
{: .no_toc }


## STEP 1. csv

```python
import numpy as np
import pandas as pd

# 데이터 경로
data_path = '/content//'

# 훈련, 검증, 테스트 데이터 경로 설정
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sampleSubmission.csv')

# df 크기 보기
train.shape, test.shape, submission.shape

# 간략 보기
train.head(3)
test.head(3)
submission.head(3)

# 열 결측값 및 데이터 타입 확인
train.info()
test.info()
```


* 피쳐 요약표

```
def resumetable(df):
  print(f"데이터셋 형상: {df.shape}")
  summary = pd.DataFrame(df.dtypes, columns=['데이터 타입'])
  summary = summary.reset_index()
  summary = summary.rename(columns={'index': '피쳐'})
  summary['결측값 갯수'] = df.isnull().sum().values
  summary['고유값 개수'] = df.nunique().values
  summary['첫 번째 값'] = df.loc[0].values
  summary['두 번째 값'] = df.loc[1].values
  summary['세 번째 값'] = df.loc[2].values

  return summary
```
