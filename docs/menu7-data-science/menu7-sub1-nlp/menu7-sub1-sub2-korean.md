---
layout: default
title: Korean
parent: NLP
grand_parent: Data Science
nav_order: 2
---

# Korean
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
## STEP 1. KoNLPy Install
* [KoNLPy official site]

<br>



<!------------------------------------ STEP ------------------------------------>
## STEP 2. example

* [자연어 처리 전처리 이해하기]

```python
### Get data
sample_data = data[:100] # 임의로 100개만 저장

### Refine
sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 한글과 공백을 제외하고 모두 제거 
sample_data[:10]

### Define stopwords
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

### Korean morphological analysis using mecab
tokenizer = Mecab() 
tokenized=[] 
for sentence in sample_data['document']: 
	temp = tokenizer.morphs(sentence) 						# tokenizer 
	temp = [word for word in temp if not word in stopwords] # Remove stopwords
	tokenized.append(temp)
	
### Word frequency
vocab = FreqDist(np.hstack(tokenized)) 
print('단어 집합의 크기 : {}'.format(len(vocab)))

### Extract top frequency words
vocab_size = 500   
vocab = vocab.most_common(vocab_size) # 상위 vocab_size개의 단어만 보존
print('단어 집합의 크기 : {}'.format(len(vocab)))

```











---

[KoNLPy official site]: https://konlpy.org/ko/latest/
[자연어 처리 전처리 이해하기]: https://wikidocs.net/64517
