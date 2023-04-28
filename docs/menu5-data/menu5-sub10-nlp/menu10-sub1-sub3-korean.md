---
layout: default
title: Korean
parent: NLP
grand_parent: Data
nav_order: 3
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
## STEP 2. 뛰어쓰기 교정

### Step 2-1. py-hanspell

```bash
# install
$ pip install py-hanspell
```

[pip install 실패시]


```python
from hanspell import spell_checker

text = "아버지가방에들어가신다나는오늘코딩을했다"

hanspell_sent = spell_checker.check(text)
print(hanspell_sent.checked)
```

### Step 2-2. 다른 교정 도구

* [soyspacing]: 형태소 분석, 품사 판별, 띄어쓰기 교정 모듈 등을 제공하는 soynlp의 띄어쓰기 교정 모듈입니다. 이 띄어쓰기 교정 모듈은 대량의 코퍼스에서 띄어쓰기 패턴을 학습한 모델을 생성 한 후, 학습한 모델을 통해 패턴대로 띄어쓰기를 교정합니다.
* [PyKoSpacinb]: 전희원님이 개발한 띄어쓰기 교정기입니다. 대용량 코퍼스를 학습하여 만들어진 띄어쓰기 딥러닝 모델로 뛰어난 성능을 가지고 있습니다.

<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 3. 형태소 분석기

### Step 3-1. KoNLPy

```bash
# install
$ pip install --upgrade pip
$ pip install konlpy
```

```python
from konlpy.tag import Kkma, Komoran, Okt, Hannanum, Mecab # Mecab은 Windows 미지원

kma = Kkma()
komoran = Komoran()
okt = Okt()
hannanum = Hannanum()

text = "아버지가 방에 들어가신다 나는 오늘 코딩을 했다"

print("Kkma: ", kkma.morphs(text))
print("Komoran: ", komoran.morphs(text))
print("Okt: ", okt.morphs(text))
print("Hannanum: ", hannanum.morphs(text))
```

### Step 3-2. 다른 분석 도구

* soynlp: soynlp에서는 L tokenizer, MaxScoreTokenizer와 같은 형태소 분석기도 제공하고 있습니다. 형태소 분석기 외에도 명사 추출기 등 한국어 자연어 분석을 위한 다양한 기능 존재
* Khaiii: 2018년에 카카오가 공개한 오픈소스 한국어 형태소 분석기
* Google sentencepiece: 2018년에 구글에서 공개한 형태소 분석 패키지 

<br>

<!------------------------------------ STEP ------------------------------------>

### STEP 4. 한글 데이터

* [KorQuAD 데이터]
* [네이버 영화 리뷰 데이터]
* [한국어 위키 백과] : 크롤링 필요


<br>

<!------------------------------------ STEP ------------------------------------>
## STEP 5. Example

### Step 5-1.
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

<br>

### Step 5-2. 명사 추출 및 빈도 계산
* [명사 추출 및 빈도 계산]

<br>

---

[KoNLPy official site]: https://konlpy.org/ko/latest/
[자연어 처리 전처리 이해하기]: https://wikidocs.net/64517
[명사 추출 및 빈도 계산]: https://liveyourit.tistory.com/57
[pip install 실패시]: https://www.codeit.kr/tutorials/42/py-hanspell-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0
[soyspacing]: https://github.com/lovit/soynlp
[PyKoSpacinb]: https://github.com/haven-jeon/PyKoSpacing

[KorQuAD 데이터]: https://korquad.github.io/
[네이버 영화 리뷰 데이터]: https://github.com/e9t/nsmc/
[한국어 위키 백과]: https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8C%80%EB%AC%B8