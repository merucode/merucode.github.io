---
layout: default
title: Linux CMD
parent: Linux
grand_parent: Etc
nav_order: 1
---

# Linux CMD

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

## STEP 1. Basic CMD

```bash
### 현재 디렉토리 (working directory)의 경로를 출력
$ pwd


### PATH 경로에 해당하는 디렉토리로 이동
$ cd [PATH]
$ cd dir1
$ cd path/to/foo    # foo 디렉토리로 이동
$ cd path/to/bar    # bar 디렉토리로 이동
$ cd -              # 다시 foo 디렉토리로 이동


### PATH에 해당하는 디렉토리/파일 정보 출력. PATH 아규먼트를 안 주면 현재 디렉토리 정보 출력
$ ls [-al] [PATH]
$ ls -al dir1
$ ls -al
# a 옵션: 숨겨진 파일을 포함한 모든 파일을 보여줍니다. 
# l 옵션: long format을 사용해서 더 자세한 정보를 보여줍니다


### 디렉토리를 생성
$ mkdir PATH1 PATH2 ...		# 여러 디렉토리 생성 가능
$ mkdir dir2
$ mkdir -p folder/sub_folder
# p 옵션: 계층형 디렉토리 생성


### 파일 생성
$ touch PATH1 PATH2 ...
$ touch file1.txt


### vim 텍스트 에디터 실행
$ vim [PATH]
$ vim file1.txt


### 파일 내용 출력
$ cat PATH1 PATH2 ...
$ cat file1.txt


### 파일 첫/마지막 내용 출력
$ head [-n count] PATH
$ head -n 5 file1.txt
$ tail [-n count] PATH
$ tail -n 5 file1.txt


### 파일 내용 페이지 단위로 나눠서 출력
$ less PATH
$ less file1.txt
# 줄 이동: 위쪽, 아래쪽 방향키	/ 다음 페이지: space 아니면 f / 이전 페이지: b / 마지막 페이지: G / 처음 페이지: g


### 디렉토리/파일을 이동하거나 이름을 변경
# DEST_PATH가 존재하는 디렉토리의 경로일 경우 파일 이동
# 그렇지 않으면 DEST_PATH로 이름 변경
$ mv [-i] SOURCE_PATH DEST_PATH
$ mv -i file1.txt file2.txt		# 파일 이름 변경
$ mv -i file1.txt dir1			# 파일 이동
# i 옵션: 똑같은 이름의 디렉토리/파일 있을 경우 확인하면서 수행


### 파일 복사
# DEST_PATH가 존재하는 디렉토리의 경로일 경우 그 안으로 파일 복사 
# 그렇치 않으면 DEST_PATH 이름으로 복사
$ cp [-ri] SOURCE_PATH DEST_PATH
$ cp -i file1.txt file2.txt
$ cp -ri dir1 dir2
# r 옵션: 디렉토리 복사시 사용
# i 옵션: 똑같은 이름의 디렉토리/파일 있을 경우 확인하면서 수행

### 파일 삭제
$ rm [-rif] PATH1 PATH2 ...
$ rm file1.txt file2.txt
$ rm -rf dir1
# r 옵션: 디렉토리 삭제시 사용 
# f 옵션: 확인하지 않고 바로 삭제
```



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 9. Note

```bash
$ echo

### 새로운 그룹을 시스템에 추가(root만 가능)
$ addgroup [OPTION] [GROUP]


### 사용자 계정을 추가(root만 가능)
```

