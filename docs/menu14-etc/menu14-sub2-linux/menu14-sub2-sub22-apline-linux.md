---
layout: default
title: Apline Linux
parent: Linux
grand_parent: Etc
nav_order: 22
---

# Apline Linux
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


도커쓰다보면 알파인을 리눅스를 많이 봅니다. 왜 그러죠?

한줄요약 = 용량이 아주 작습니다. 대단한게 아니라 그게 답입니다 ㅋ

 

용량작은건 그렇다 치고 뭐가 다른가요?

1. 패키지 인스톨러가 apk이다.. 전통적인 yum, apt가 아님

2. 기본 쉘이 배쉬가 아니라 애쉬다(ash)

3. 시스템 구동에 필요하지 않은 편의성을 위한 커맨드는 전혀 들어있지 않음...

(저용량이라고 다 좋은게 아니고 결국 필요한거 다 깔고나면 용량 늘죠...)

 

고로 그렇기 때문에 도커컨테이너로서 많이 쓰는것이다!

 

자 한번 알아봅시다.

 

컨테이너 인스톨/로그인

docker run -it alpine ash
습관적으로 독커런할 때 bash붙이시는 분도 계실텐데 배쉬가 아니라 ash!

당연히 배쉬쉘이려니 하고 도커 로그인 했는데 에러나서 이 컨테이너 이상한가 보다 하신분?

알파인 아니었던감? 그럴 때는 ash한번 해보시거나 이도 저도 안 되면 sh로 로그인 해보시기를~

 

알파인의 패키지 관리

# 패키지 확인
apk info

# 패키지 업데이트
apk update

# 패키지 검색
apk search [패키지명]

# 패키지 설치
apk add [패키지명]

# 패키지 업데이트 & 설치
apk --update add

# apk 캐쉬삭제
rm -rf /var/cache/apk/*
뭐 그래봐야 레드햇계열이나 데비안 계열이려니 하고

언제나 처럼

yum, apt를 날려봐도 묵묵 부답인 녀석입니다. apk하세요~

 

알파인으로 데스크톱 개발환경을 꾸리겠다도 아닐테고

도커 컨테이너 빌드가 목적이라 가정하면... (뭐 아니신 분은 제가 딱히 드릴 말씀이...)

 

이것만 유의하면 될 겁니다!

<br>

<hr>

### 관련 사이트
>* [docker의 절친 alpine Linux 3분 요약!](https://devcheat.tistory.com/5)
