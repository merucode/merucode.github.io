---
layout: default
title: Pipenv
parent: Virtual Environment
grand_parent: Python
nav_order: 6
---

# Pipenv
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
# STEP 1. pipenv 소개

**pipenv**는  [파이썬에서 공식으로 권장하는 패키지 관리 툴](https://packaging.python.org/en/latest/tutorials/managing-dependencies/#managing-dependencies)

###  Step1-1. pipenv의 특징
  
 1) 패키지 설치와 가상 환경 사용을 위해 pip와 가상 환경 모듈을 따로 쓸 필요가 없다.  
 2) 패키지 기록 파일(e.g., venv 또는 virtualenv의 requirement.txt)로 Pipfile과 Pipfile.lock을 사용한다.  
 3) lock 파일의 자동 Hash(=텍스트를 임의의 암호화된 텍스트로 변환)로 안전한 버전 관리가 가능하다.  
4) Dependency Graph 시각화 기능 제공(명령어: pipenv graph)  
5) 패키지를 설치하면 자동으로 Pipfile 파일에 변경사항이 반영된다.

<br>

# STEP 2. pipenv 설치
### Step 2-1. windosw pipenv 설치

```bash
$ pip install pipenv
```

<br>

### Step 2-2. ubuntu pipenv 설치 
* **pip install**

	```bash
	$ sudo apt install python3-pip
	```

* **경로 설정** :  pipenv 설치 시  `~/.local/bin`에 실행파일 만듬. ubuntu에선 이 경로가 실행 경로에 잡혀있지 않기에, 수동으로 추가

	```bash
	# 현재 PATH 확인
	echo $PATH
	# PATH 경로 추가
	PYTHON_BIN_PATH="$(python3 -m site --user-base)/bin"
	PATH="$PATH:$PYTHON_BIN_PATH"
	```
	
* **pipenv 설치**

	```bash
	$ pip3 install --user pipenv
	```

<br>

# STEP 3. 가상환경 생성/삭제
### Step 3-1.  가상환경 폴더 생성

```bash
$ mkdir 폴더이름 && cd 폴더이름
```

<br>

### Step 3-2. 가상환경 생성

```bash
# 사용할 python 버전 기재
$ pipenv --python 3.x
```

<br>

### Step 3-3. 가상환경 삭제

```bash
# 삭제하고자 하는 가상환경 폴더에서 실행
$ pipenv --rm
```

<br>

# STEP 4. 가상환경 활성화/비활성화
### Step 4-1. 가상환경 활성화
```bash
$ pipenv shell
```
<br>

### Step 4-2. 가상환경 비활성화
```bash
$ exit
```

<br>

# STEP 5. 가상환경 패키지 설치
### Step 5-1. 패키지 설치
```bash
# 가상환경 활성화 상태에서
# 개발용 패키지 설치는 —dev 옵션을 추가
$ pipenv install 패키지명
```

<br>

### Step 5-2. 설치된 패키지 확인
```bash
$ pipenv graph
```

<br>

# STEP 6. pipfile
### Step 6-1. pipfile
* Pipfile은 해당 가상환경에 설치된 파이썬 버전, 패키지별 이름과 버전을 기록
* pipenv 가상환경을 설치했다면 해당은 기본적으로 설치

<br>

### Step 6-2. pipfile.lock
*  Pipfile.lock 파일을 열어보시면 Pipfile과 다르게 텍스트가 암호화 됨

<br>

<hr>

### 관련 사이트
 * [ [Python] pipenv 등장배경, 설치, 패키지 관리 방법](https://heytech.tistory.com/320)
* [[Python] Jupyter Notebook(Lab)에 가상환경 커널 추가 방법](https://heytech.tistory.com/324)
