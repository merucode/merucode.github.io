---
layout: default
title: NLP Basic
parent: NLP
grand_parent: Data
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

<!------------------------------------ STEP ------------------------------------>
## STEP 1. NLP

- Preprocessing
	1. Tokenize : 자연어 데이터를 분석을 위한 작은 단위(토큰)로 분리
	2. Cleaning : 분석에 큰 의미가 없는 데이터들을 제거
	3. Normalization : 표현 방법이 다르지만 의미가 같은 단어들을 통합
	4. Integer encoding : 컴퓨터가 이해하기 쉽도록 자연어 데이터에 정수 인덱스를 부여
	
- Terms
	- Corpus : language data set

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 2.  Word Preprocessing
### Step 2-1. Nltk package Install
* [nltk official site](https://www.nltk.org/install.html)

<br>

### Step 2-2. Word Tokenization

```python
from nltk.tokenize import word_tokenize
import nltk 
nltk.download('punkt')	# 마침표 등을 고려하여 토큰화 가능케 해줌

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

<!------------------------------------ STEP ------------------------------------>
## STEP 3. Sentence Preprocessing
### Step 3-1. Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
text = "My email address is 'abcde@codeit.com'. Send it to Mr.Kim."

# 문장 토큰화
tokenized_sents = sent_tokenize(text)
```

<br>

### Step 3-2. POS(Part of Speech Tagging)

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# 품사 태깅 함수
def pos_tagger(tokenized_sents):
    pos_tagged_words = []

    for sentence in tokenized_sents:
        # 단어 토큰화
        tokenized_words = word_tokenize(sentence)
    
        # 품사 태깅
        pos_tagged = pos_tag(tokenized_words)
        pos_tagged_words.extend(pos_tagged)
    
    return pos_tagged_words
```

* pos_tag
[image](https://www.codeit.kr/learn/5890)

<br>

### Step 3-3. Lemmatization(표제어 추출)

* Lemma : 단어의 표준적 어원
	* ex) am, is, are → be
* WordNet : Large english word dictionary from NLTK
* use `WordNet POS Tag` : it is need to convert as `pos_tag` → `WordNet POS Tag`

|WorldNet POS Tag|POS|
|--|--|
|n (wn.NOUN)|noun|
|a (wn.ADJ)|adjective|
|r (wn.ADV)|adverb|
|v (wn.VERB)|verb|

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')

### `pos_tag` → `WordNet POS Tag`
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB

### Lemmatization
def words_lemmatizer(pos_tagged_words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []

    for word, tag in pos_tagged_words:
		# WordNet Pos Tag로 변환
        wn_tag = penn_to_wn(tag)	

		# 품사를 기준으로 표제어 추출
        if wn_tag in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
            lemmatized_words.append(lemmatizer.lemmatize(word, wn_tag))
        else:
            lemmatized_words.append(word)

    return lemmatized_words

```

<br>

### Step 3-4. NLP Example

```python
### Read data
df = pd.read_csv('imdb.tsv', delimiter = "\\t")

### Convert to small letter
df['review'] = df['review'].str.lower()

### Sentence tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

df['sent_tokens'] = df['review'].apply(sent_tokenize)

### POS tagging
from preprocess import pos_tagger # Step 3-2
df['pos_tagged_tokens'] = df['sent_tokens'].apply(pos_tagger)

### Lemmatization
from preprocess import words_lemmatizer # Step 3-3
df['lemmatized_tokens'] = df['pos_tagged_tokens'].apply(words_lemmatizer)

### Additional preprocessing
from preprocess import clean_by_freq, clean_by_len, clean_by_stopwords # STEP 2

stopwords_set = set(stopwords.words('english')) # Step 2-4

df['cleaned_tokens'] = df['lemmatized_tokens'].apply(lambda x: clean_by_freq(x, 1)) # Step 2-3
df['cleaned_tokens'] = df['cleaned_tokens'].apply(lambda x: clean_by_len(x, 2)) # Step 2-3
df['cleaned_tokens'] = df['cleaned_tokens'].apply(lambda x: clean_by_stopwords(x, stopwords_set)) # Step 2-4

df[['cleaned_tokens']]	### Token Reselut

### Merge data
def combine(sentence):
    return ' '.join(sentence)

df['combined_corpus'] = df['cleaned_tokens'].apply(combine)

df[['combined_corpus']]	### Corpus Reselut
```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 4. Integer Encoding

### Step 4-1. Integer Encoding

* **Interger encoding** : Mapping some integer to tokenized data for use by computers
* Integer encoding prevents further preprocessing. So After All preprocessing done, you have to do interger encoding

```python
### Preprocess token data
df[['cleaned_tokens']]

### Change token df to token list
tokens = sum(df['cleaned_tokens'], [])

### Indexing by frequency
word_to_idx = {}
i = 0

vocab = Counter(tokens)
vocab = vocab.most_common()

for (word, frequency) in vocab:
    i = i + 1
    word_to_idx[word] = i

### Encoding with above word_to_idx 
def idx_encoder(tokens, word_to_idx):
    encoded_idx = []
    
    for token in tokens:
        idx = word_to_idx[token]
        encoded_idx.append(idx)
        
    return encoded_idx

df['integer_encoded'] = df['cleaned_tokens'].apply(lambda x: idx_encoder(x, word_to_idx))
df[['integer_encoded']]
```

<br>

### Step 4-2. Padding

* **Padding** : Form a matrix by matching the lengths of different sentences to each other

```python
### max token number
max_len = max(len(item) for item in df['integer_encoded'])

### Zero Padding
for tokens in df['integer_encoded']:
    while len(tokens) < max_len:
        tokens.append(0)

### Padding result(Matrix form)
df[['integer_encoded']]	
```

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 5. Sentiment Analysis
### Step 5-1. Sentiment analysis

* **Rule-based sentiment analysis**(Topics covered this time)
* **Machine learning-based sentiment analysis**

<br>

### Step 5-2. SentiWordNet

* **Synset** : Sets of Cognitive Synonyms
* `swn.senti_synsets`
	* 긍정 감성 지수 함수: pos_score()
	* 부정 감성 지수 함수: neg_score()
	* 객관성 지수 함수: obj_score()

```python
from nltk.corpus import sentiwordnet as swn

### Setting word
happy_sentisynsets = list(swn.senti_synsets('happy'))

### Scoring sentiment → Only use happy synset 0
pos_score = happy_sentisynsets[0].pos_score()
neg_score = happy_sentisynsets[0].neg_score()
obj_score = happy_sentisynsets[0].obj_score()

### Result
print(pos_score - neg_score)
```

```python
def swn_polarity(pos_tagged_words):
    senti_score = 0

    for word, tag in pos_tagged_words:
        # PennTreeBank 기준 품사를 WordNet 기준 품사로 변경
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
            continue
    
        # Synset 확인, 어휘 사전에 없을 경우에는 스킵
        if not wn.synsets(word, wn_tag):
            continue
        else:
            synsets = wn.synsets(word, wn_tag)
    
        # SentiSynset 확인
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        # 감성 지수 계산
        word_senti_score = (swn_synset.pos_score() - swn_synset.neg_score())
        senti_score += word_senti_score

    return senti_score
```

<br>

### Step 5-3. VADER

* VADER : Difference from SentiWordNet is that the emotional index can be extracted considering abbreviations and symbols

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

### Setting Dataframe
df[['review']]

### VADER
def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    
    # VADER 감성 분석
    senti_score = analyzer.polarity_scores(text)['compound'] # neg, neu, pos, compound
    
    return senti_score

df['vader_sentiment'] = df['review'].apply(vader_sentiment)
df[['review', 'vader_sentiment']]
```
<br>