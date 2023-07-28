---
layout: default
title: Prompt Engineering
parent: Prompt
grand_parent: ML Framework
nav_order: 2
---

# Prompt Engineering
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

## STEP 0. Reference Site

* [[youtube]-빵상의 개발도상국-ChatGPT Prompt Engineering](https://www.youtube.com/watch?v=LGNVFnxDm0Y)


## STEP 1. Guideline(지침)
### Step 1-1. 명확하고 구체적인 지침 작성하기
* 전략 1-1. 구분 기호를 사용하여 구분되는 부분을 명확하게 표기
	* `(백틱), """(쌍따옴표 3개), <>,  :  등
```
백틱 세 개로 구분된 텍스트를 한 문장으로 요약합니다.
```텍스트```
```

* 전략 1-2. 구조화된 응답 요청
	
```
저자 및 장르와 함께 세 개의 가상 도서 목록을 생성합니다. book_id, title, author, genre 키와 함께 JSON 형식으로 작성해주세요.
```

* 전략 1-3. 모델에게 조건이 충족되는지 확인하도록 요청

```
큰따옴표로 구분된 텍스트가 제공됩니다.
일련의 지침이 포함된 경우 다음 형식으로 해당 지침을 다시 작성합니다:

1단계-...
2단계-...
...
N단계-...

텍스트에 일련의 지침이 포함되어 있지 않은 경우에는 "제공된 단계 없음"이라고 간단히 작성합니다.

"""차 한잔을 만드는 것은..."""
```

* 전략 1-4. "Few-shot(예시)" 프롬프팅

```
당신의 임무는 일관된 스타일로 대답하는 것입니다.

<어린이>: 인내심에 대해 가르쳐주세요.
<조부모>: 가장 깊은 계곡을 깍아내는 강은 겸손한 샘에서 흐르고, 가장 웅장한 교향곡은 한 음에서 시작되며, 가장 복잡한 직조는 하나의 실에서 시작된단다.
<어린이>: 강인함에 대해 알려주세요.
```

### Step 1-2. 모델에게 "생각할 시간" 주기

## 2. Iterative

## 3. Summarizing

## 4. Inferring
 
## 5. Transforming

## STEP 6. Expanding