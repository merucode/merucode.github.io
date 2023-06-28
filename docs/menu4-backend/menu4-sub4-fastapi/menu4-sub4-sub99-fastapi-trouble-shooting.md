---
layout: default
title: FastApi Trouble Shooting
parent: FastApi
grand_parent: Backend
nav_order: 99
---

# FastApi Trouble Shooting
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

## STEP 1. JSONResponse 사용시 `TypeError: Object of type date is not JSON serializable` 에러 발생

### Steop 1-1. Trouble

* JSONResponse를 이용해 응답시 `TypeError: Object of type date is not JSON serializable` 에러 발생
	
	```
		...
		return JSONResponse((content="data":response_dict_lst}))
	```

### Step 1-2. Casue

* [https://louky0714.tistory.com/148](https://louky0714.tistory.com/148)
* 응답 데이터(예제 상 `response_dict_lst`)내에 date함수를 사용해서 발생하는 문제. 소스 뿐만 아니라 json data내에 날짜 변수로 된 data가 있을 경우에도 동일하게 발생

### Step 1-3. Solution

* `datetime.datetime.strftime(data, "%Y-%m-%d)`로 문자열 변환



<br>
