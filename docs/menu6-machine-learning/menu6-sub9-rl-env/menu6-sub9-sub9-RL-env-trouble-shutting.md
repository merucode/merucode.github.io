---
layout: default
title: RL ENV Trouble Shooting
parent: RL ENV
grand_parent: Machine Leaning
nav_order: 9
---

# Summary
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

## STEP. `NameNotFound: Environment snake doesn't exist.`

### Step. Trouble

* Colab에서 Custom ENV 설치 후 gym.make 실행 시 오류 발생

```bash
!pip insatll -e snake
```

```python
...
env = gym.make(...)
```

### Step. Cause

### Step. Solution

* 런타임 해제 후 다시 연결

<br>

###

* [Colab에서 랜더링](https://medium.com/analytics-vidhya/rendering-openai-gym-environments-in-google-colab-9df4e7d6f99f) 
* [?](https://somjang.tistory.com/entry/Google-Colab%EC%97%90%EC%84%9C-OpenAI-gym-render-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95)