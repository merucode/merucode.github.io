---
layout: default
title: Conda
parent: Virtual Environment
grand_parent: Python
nav_order: 5
---

# Conda
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



### 1. Aaconda 최신 버전으로 업데이트

가상환경 생성전에 Anaconda를 먼저 최신 버전으로 업데이트 합니다.

```
conda update conda
```

### 2. 설치된 가상 환경 목록 출력

```
conda env list

# conda environments:#
base                  *  C:\\Anaconda3
```

- 표시는 현재 활성화된 가상환경을 의미

### 3. 새로운 가상 환경 생성

python 3.8 버전의 새로운 가상 환경을 생성합니다.

```
conda create --name [가상환경이름] python=3.8
conda create --name mypython python=3.8
```

### 4. 설치된 가상 환경 활성화 하기

```
conda activate [가상환경이름]
# To activate this environment, use##     
$ conda activate mypython
```

`conda env list`로 생성된 가상환경 리스트를 확인하고 *표시가 되어 있는 환경이 활성화된 가성환경입니다.

### [참고] 가상환경 비활성화 하기

```
# To deactivate an active environment, use##     
$ conda deactivate
```

### 5.  가상환경 삭제

```
conda env remove -n [가상환경이름]
conda env remove -n 가상환경이름
```

### 6.  가상환경에 jupyter notebook 설치하기

```jsx
pip install jupyter notebook
pip install ipykernel
```

jupyter notebook 만 설치하면 ipykernel이 자동으로 다운받아집니다. 혹시나 안 받아진다면 ipykernel을 따로 다운해주도록 합니다.

### 7.  가상환경에 kernel 연결하기

```
python -m ipykernel install --user --name 가상머신이름 --display-name [표시할이름]
python -m ipykernel install --user --name 가상머신이름 --display-name "표시할이름"
```

### 8. 가상환경 복사하기

```
conda create -n [가상환경이름] --clone [복제가상환경이름]
conda create -n myconda_copy --clone myconda
```

### 9. 가상환경 설치 목록 보기

(가상환경 활성화) `conda list`

```jsx
(temanet) C:\\Windows\\System32>conda list
```

### 10. 가상환경 내 패키지 설치하기

```
conda install -n [환경이름] [패키지이름]
(temanet) C:\\Windows\\System32>conda install -n temanet django
```