---
layout: default
title: Linux CMD
parent: Linux
grand_parent: Etc
nav_order: 2
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
$ adduser [OPTION] [user]
# s : 사용자 생성 시 사용자가 사용할 셸을 지정한다.
# g : 그룹을 지정할 때 사용하는데, 지정할 그룹이 미리 생성 필요
# G : 기본 그룹 이외에 추가로 그룹에 속하게 할 경우에 쓴다.
# useradd와 차이(ubuntu)
    # useradd : 순수하게 계정만 생성, 기본 쉘 sh 할당, 홈 디렉토리나 비밀번호 따로 설정 필요
    # adduser : 계정 생성 시 사용자 정보 입력 받음. 홈 디렉토리 자동 생성


### 파일의 소유권과 그룹을 변경하는 명령어(change owner)
$ chown [OPTION] [owner][:[group]] FILE
# 소유자 변경 			: chown user FILE
# 소유자그룹 변경(. 사용) : chown .group FILE
# 둘다 변경				: chwon user:group FILE
$ chown -R user apple
# R(--recursive) : 지정한 파일의 하위까지 변경


### 파일의 권한을 변경하는 명령어(change mode)
$ chmod [OPTION] [MODE] [FILE]
## [OPTION]
# v        : 모든 파일에 대해 모드가 적용되는 진단(diagnostic) 메시지 출력.
# f        : 에러 메시지 출력하지 않음.
# c        : 기존 파일 모드가 변경되는 경우만 진단(diagnostic) 메시지 출력.
# R        : 지정한 모드를 파일과 디렉토리에 대해 재귀적으로(recursively) 적용.
## [MODE] : 파일에 적용할 모드(mode) 문자열 조합
# u,g,o,a : 소유자(u), 그룹(g), 그 외 사용자(o), 모든 사용자(a) 지정.
# +,-,=   : 현재 모드에 권한 추가(+), 현재 모드에서 권한 제거(-), 현재 모드로 권한 지정(=)
# r,w,x   : 읽기 권한(r), 쓰기 권한(w), 실행 권한(x)
# X       : "디렉토리" 또는 "실행 권한(x)이 있는 파일"에 실행 권한(x) 적용.
# s       : 실행 시 사용자 또는 그룹 ID 지정(s). "setuid", "setgid".
# t       : 공유모드에서의 제한된 삭제 플래그를 나타내는 sticky(t) bit.
# 0~7     : 8진수(octet) 형식 모드 설정 값.
$ chmod u+w FILE        # 파일 소유 사용자에게 쓰기 권한 추가.
$ chmod u=rwx FILE      # 파일 소유 사용자에게 읽기, 쓰기, 실행 권한 지정.
$ chmod u-x FILE        # 파일 소유 사용자의 실행 권한 제거.
$ chmod g+w FILE        # 파일 소유 그룹에 쓰기 권한 추가.
$ chmod -R g+x DIR      # DIR 디렉토리 하위 모든 파일 및 디렉토리에 그룹 실행(x) 권한 추가.
$ chmod a-x *           # 현재 디렉토리의 모든 파일에서 모든 사용자의 읽기 권한 제거.
```

