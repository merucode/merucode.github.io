---
layout: default
title: NLP Basic
parent: NLP
grand_parent: Data Science
nav_order: 2
---

# NLP Basic
{: .no_toc .d-inline-block }
ing
{: .label .label-green }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>


## STEP 1. NLP

<p>

- Preprocessing
	1. Tokenize : 자연어 데이터를 분석을 위한 작은 단위(토큰)로 분리
	
	2. Cleaning : 분석에 큰 의미가 없는 데이터들을 제거
	
	3. Normalization : 표현 방법이 다르지만 의미가 같은 단어들을 통합

	4. Integer encoding : 컴퓨터가 이해하기 쉽도록 자연어 데이터에 정수 인덱스를 부여
	
	   
	
- Terms
	- Corpus : language data set

<br>

## STEP 2.  Word Preprocessing
### Step 2-1. Nltk package Install
* [nltk official site](https://www.nltk.org/install.html)

<br>

### Step 2-2. Word Tokenization

```python
from nltk.tokenize import word_tokenize
import nltk 
nltk.download('punkt')	# 마침표 등을 고려하여 토큰화

text = "Although it's not a happily-ever-after ending, it is very realistic."  

### 단어 토큰화
tokenized_words = word_tokenize(text) print(tokenized_words)
```

<br>

### Step 2-3. Cleaning

```python
from collections import Counter 

### 등장 빈도 기준 정제 함수  
def clean_by_freq(tokenized_words, cut_off_count): 
	# 파이썬의 Counter 모듈을 통해 단어의 빈도수 카운트하여 단어 집합 생성 
	vocab = Counter(tokenized_words) 
	# 빈도수가 cut_off_count 이하인 단어 set 추출 
	uncommon_words = {key for key, value in vocab.items() if value <= cut_off_count} 
	# uncommon_words에 포함되지 않는 단어 리스트 생성 
	cleaned_words = [word for word in tokenized_words if word not  in uncommon_words] 
	return cleaned_words 

### 단어 길이 기준 정제 함수  
def clean_by_len(tokenized_words, cut_off_length): 
	# 길이가 cut_off_length 이하인 단어 제거 
	cleaned_by_freq_len = [] 
	for word in tokenized_words: 
		if len(word) > cut_off_length: 
			cleaned_by_freq_len.append(word) 
		return cleaned_by_freq_len
```

<br>

### Step 2-4. Remove Stopword

```python
from nltk.corpus import stopwords
import nltk

### Step 1. Setting stopword
nltk.download('stopwords')	# download stopword
stopwords_set = set(stopwords.words('english'))

stopwords_set.add('hello')	# Add stopword 
stopwords_set.remove('the')	# Remove stopword

my_stopwords_set = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves'}	# Set customized stopword


### Step 2. Def remove stopword   
def  clean_by_stopwords(tokenized_words, stop_words_set): 
	cleaned_words = []
	 
	for word in tokenized_words: 
		if word not in stop_words_set: 
			cleaned_words.append(word) 
			
	return cleaned_words
```

<br>

### Step 2-5. Normalization

```python
### if English
text = text.lower()	# lowercase conversion


# 동의어 사전 
synonym_dict = {'US':'USA', 'U.S':'USA', 'Ummm':'Umm', 'Ummmm':'Umm' } 
normalized_words = []

for word in tokenized_words: 
	# 동의어 사전에 있는 단어라면, value에 해당하는 값으로 변환  
	if word in synonym_dict.keys(): 
		word = synonym_dict[word] 
	
	normalized_words.append(word)
```

```python 
### if English, normalize pre, lize, etc 
from nltk.stem import PorterStemmer 
 
# 포터 스테머 어간 추출 함수 
def  stemming_by_porter(tokenized_words): 
	porter_stemmer = PorterStemmer() 
	porter_stemmed_words = [] 
	
	for word in tokenized_words: 
		stem = porter_stemmer.stem(word) 

	porter_stemmed_words.append(stem) 
	return porter_stemmed_words
```





<br>


## STEP 3. Sentence Preprocessing


<br>

## STEP 4. Integer Encoding


<br>

## STEP 5. Sentiment Analysis

