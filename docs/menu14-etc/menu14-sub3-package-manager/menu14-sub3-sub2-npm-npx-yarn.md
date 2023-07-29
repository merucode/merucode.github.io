---
layout: default
title: npm, npx, yarn
parent: Package Manager
grand_parent: Etc
nav_order: 3
---

# Package Manager

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

## STEP 1. npm

```python
### npm: node.js 패키지 매니저(node_modules에 패키지 설치)
"""
1. 패키지 설치
2. 버전 관리 제공
"""
# package.json의 dependencies에 있는 모든 패키지를 설치
$ npm install   
$ npm i			# npm install 줄인 명령어 	

# node_modules에 [package] 설치 
$ npm install [package] 

# [package] 설치 및 package.json의 dependencies 객체에 [package] 추가
$ npm install [package] --save 	

# dependencies가 아닌 devDepenencies 객체에 추가
npm install [package] --save -dev

# [package] 전역 설치
$ npm install [package] -g

```

## STEP 2. npx

```python
### npx: node.js 패키지를 실행시키는 하나의 도구
"""
1. 실행시킬 패키지가 로컬에 저장되어 있는지 먼저 확인
2. 존재한다면 실행
3. 존재하지 않는다면 npx가 가장 최신 버전을 설치하고 실행
4. 모듈이 많아 업데이트가 잦은 create-react-app 등의 경우 npx를 이용해 설치하는 것을 권장
"""

```


<br>

## STEP 9. 기본 용어 정리
* package.json : 모듈 설치시 자동으로 생성되는 node.js 버전 관리 파일
* node_modules : 모든 모듈의 저장공간
