---
layout: default
title: pandas Basic
parent: pandas
grand_parent: Python
nav_order: 2
---

# pandas Basic
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

## STEP 1. pandas 기본

### Step 1-1. df 생성 방법
* `df = pd.DataFrame()` 생성 후 `df['colume'] = list`
* `df = pd.DataFrame(list or array or Seies)`
* `df = pd.DataFrame(dict)`(key is column name)



### Step 1-2. index/column name 지정
* **전체** 
	* `df = pd.DataFrame(value_list, columns = ['name1','name2'])` : 생성 시
	* `df.columns = ['name1', 'name2']` : 이름 재지정
	* `df.index = ['index1', 'index2']` : 인덱스 재지정
	* `df.set_index('col1', inplace=True)` : col1으로 인덱스 재지정
		* `df['new_col'] = df.index` : 인덱스를 새로운 열로 지정 

* **개별**
	* `df.rename(columns={'old_row':'new_row'}, inplace=True)`
	* `df.index.name = "index_name"` : index column name 지정



### Step 1-3. 기본 명령어
```python
list(df.columns) # columns name list 반환
list(df.index)	 # index list 반환
df.columns[0]	 # 첫 번째 columns name 반환
df.index[0]		 # 첫 번째 index 반환
df.dtypes		 # column 별 데이터 type 확인
```



### Step 1-4. pandas data type

| dtype        | 설명            |
| ------------ | --------------- |
| `int64`      | 정수            |
| `float64`    | 소수            |
| `object`     | 텍스트          |
| `bool`       | 불린(참과 거짓) |
| `datetime64` | 날짜와 시간     |
| `category`   | 카테고리        |



<br>



## STEP 2. DataFrame Data Read
### Step 2-1. read_csv

```python
df = pd.read_csv('path/data.csv')							# index 자동 지정, 첫번째 행 제목 지정
df = pd.read_csv('path/data.csv', header=None)				# 제목 지정 없음
df = pd.read_csv('path/data.csv', index_col='column_name')	# 인덱스 열 지정
```



<br>



## STEP 3. DataFrame 인덱싱

### Step 3-1.  **이름**

| 이름으로 인덱싱하기           | 기본 형태                             | 단축 형태                      |
| ----------------------------- | ------------------------------------- | ------------------------------ |
| 하나의 row 이름               | `df.loc["row4"]`                      |                                |
| row 이름의 리스트             | `df.loc[["row4", "row5", "row3"]]`    |                                |
| row 이름의 리스트 슬라이싱    | `df.loc["row2":"row5"]`               | `df["row2":"row5"]`            |
| 하나의 column 이름            | `df.loc[:, "col1"]`                   | `df["col1"]`                   |
| column 이름의 리스트          | `df.loc[:, ["col4", "col6", "col3"]]` | `df[["col4", "col6", "col3"]]` |
| column 이름의 리스트 슬라이싱 | `df.loc[:, "col2":"col5"]`            |                                |

* 1 줄 인덱싱의 경우 Series로 반환

  

### Step 3-2. **위치**

| 위치로 인덱싱하기             | 기본 형태               | 단축 형태 |
| ----------------------------- | ----------------------- | --------- |
| 하나의 row 위치               | `df.iloc[8]`            |           |
| row 위치의 리스트             | `df.iloc[[4, 5, 3]]`    |           |
| row 위치의 리스트 슬라이싱    | `df.iloc[2:5]`          | `df[2:5]` |
| 하나의 column 위치            | `df.iloc[:, 3]`         |           |
| column 위치의 리스트          | `df.iloc[:, [3, 5, 6]]` |           |
| column 위치의 리스트 슬라이싱 | `df.iloc[:, 3:7]`       |           |

* 1 줄 인덱싱의 경우 Series로 반환

  

<br>

## STEP 4. Data Frame 조건 인덱싱
### Step 4-1. df boolean
```python
value_list = [[200,6],[800,9]]
df = pd.DataFrame(value_list, columns=['price','count'])
#   price count
# 0  200 	6 
# 1  800 	9

df = df.loc[:,[True,False]]
#   price 
# 0  200
# 1  800

df['price'] > 500
# 0 False 
# 1 True

df[df > 500] = 0
#   price count
# 0  200 	6 
# 1   0 	9
```
* 개수에 맞는 리스트 요소를 모두 적어줘야 함

  

### Step 4-2. 조건 인덱싱
```python
# AND &, OR |

df.loc[df['price']>500]	# 열조건 → 인덱스와 불린 값 반환을 사용하여 해당 열 추출 가능
#	price count
# 1	 800	 9

df = df.loc[df['price'] > df['count']]	# df 열끼리도 비교 가능

df.loc[(df['price']>500) & (df['count']==9)] # AND 조건
#	price count
# 1	 800	 9
```

```python
df[df.loc[:, 'B':'E'] < 80] = 0
df[df.loc[:, 'B':'E'] >= 80] = 1
# 'B':'E' 열 중에 80 이상은 1, 80 미만은 0으로 처리됨
# 즉, 조건에 따라 True인 요소들만 처리 됨
```



<br>



## STEP 5. DataFrame 값 변경/추가/삭제

### Step 5-1. df 값 변경

```python
# 한 요소 변경
df.loc['index','col1'] = "new_value"
df.loc[1,2] = "new value"

# raw or column value 변경
df.loc['index'] = new_value_list # loc or iloc로 열 혹은 행 지정 후 값 리스트 입력
df.loc['index'] = "Yes"			 # index에 해당하는 raw value 모두 Yes 변경

# 여러 줄 변경
df[['col1','col2']] = "Yes"		# col1, col2 value 모두 Yes 변경

# 조건 변경
df.loc[df['price'] > 500] = "Yes"	# price 500 넘는 raw value 모두 Yes 변경 
```

<br>

### Step 5-2. df 값 추가

```python
# raw 추가
df.loc['raw_new'] = new_value_list
df.loc['raw_new'] = "Yes"	# value 모두 Yes인 raw_new 추가

# column 추가
df['col_new'] = new_value_list
df['col_new'] = "Yes"	# value 모두 Yes인 col_new 추가
```

<br>

### Step 5-3. df 값 제거

```python
# raw 제거(axis=0 가능)
df.drop('row1', axis="index", inplace="True") # inplace="False" : 기존 df 영향 X

# column 제거(axis=1 가능)
df.drop('col1', axis="column", inplace="True")

# 여러 줄 삭제
df.drop(['row1','row2'], axis="index", inplace="True")
```



<br>



## STEP 6. 큰 DataFrame 처리하기

### Step 6-1. df 살펴보기
```python
df.head(5)	# 첫 5줄
df.tail(5)	# 아래 5줄

df.shape		# (row, column)
df.columns		# columns 값 
df.info()		# columns data 갯수, 형식
df.describe() 	# df 통계값

df.sort_values(by='col1')					# col1 오름차순 정렬
df.sort_values(by='col1', ascending=False)	# col1 내림차순 정렬
df.sort_values(by='col1', inplace=True)		# df 자체 변환
```

### Step 6-2. Series 살펴보기

* **df에서 열을 추출하면 Series가 됨**

```python
se.unique()
se.value_counts()
se.dexcribe()

df['col1'].unique()	    	# 중복을 제외한 value들의 array 반환
df['col1'].value_counts()	# 각 value 별로 몇 개가 있는지 series 반환
df['col1'].describe()		# series 통계
```

* **DataFrame에서 특정 column의 중복 갯수 조건을 이용할 때**

	```python
	# df에서 col하나를 추출(Series)하여 value 중복 값 확인
	# 확인된 value 중복 값 중 특정 조건을 만족하는 것들 list 반환
	# for문을 통해 df에서 해당 list 종목들 값 처리
	
	df_counts = df['course name'].value_counts()
	df_values = list(df_counts[df_counts < 5].index) # value 갯수가 5개 이하인 value list 반환
	# 위 2줄을 한줄로 가능
	# df_values = list(df_counts[df['course name'].value_counts() < 5].index)
	
	for i in df_values: 
		df.loc[df["course name"] == i, "status"] = "not allowed"
	```

<br>

### STEP 7. ETC

```python
df.T  # Convert rows and columns
```
