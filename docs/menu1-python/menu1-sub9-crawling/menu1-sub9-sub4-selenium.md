---
layout: default
title: Selenium
parent: Crawling
grand_parent: Python
nav_order: 2
---

# Selenium
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

## STEP 0. Reference

* [Selenium(with docker)](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub21-selenium.html)
* [[Selenium] Ubuntu 20.04에 Chrome driver 설치하기](https://velog.io/@choi-yh/Ubuntu-20.04-selinium-%EC%84%B8%ED%8C%85)

<br>

## STEP 1. 

```python
from selenium import webdriver  # selenium의 webdriver를 사용
from selenium.webdriver.common.keys import Keys # selenium으로 키를 조작
import time                     # 페이지 로딩 wating

# DevToolsActivePort file doesn't exist error 해결을 위한 options 추가
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument("--single-process")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options) 

driver.get('https://www.google.co.kr/')

time.sleep(3)

print(driver.current_url)
driver.close
```

* https://hyunsooworld.tistory.com/entry/%EC%85%80%EB%A0%88%EB%8B%88%EC%9B%80-%EC%98%A4%EB%A5%98-AttributeError-WebDriver-object-has-no-attribute-findelementbycssselector-%EC%98%A4%EB%A5%98%ED%95%B4%EA%B2%B0
* https://sualchi.tistory.com/13721870
