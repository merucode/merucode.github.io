---
layout: default
title: Linux Basic
parent: Linux
grand_parent: Etc
nav_order: 3
---

# Linux Basic

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

## STEP 5. Vim

### Step 5-1. Vim mode and change mode

| Vim mode                                                     | Change mode                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20230429085509873](./../../../images/menu14-sub2-sub3-linux-basic/image-20230429085509873.png) | ![image-20230429085523721](./../../../images/menu14-sub2-sub3-linux-basic/image-20230429085523721.png) |



### Step 5-2. Hotkey

| Description                   | Hotkey                                                       |
| ----------------------------- | ------------------------------------------------------------ |
| 텍스트 한 줄 복사             | 일반 모드 → 복사하고 싶은 줄에 커서 위치 → yy                |
| 텍스트 한 줄 잘라내기         | 일반 모드 → 잘라내고 싶은 줄에 커서 위치 → dd                |
| 특정 영역 복사                | 비주얼 모드(V는 줄 단위, v는 글자 단위) → 복사하고 싶은 영역 커서로 설정 → y |
| 특정 영역 잘라내기            | 비주얼 모드(V는 줄 단위, v는 글자 단위) → 잘라내고 싶은 영역 커서로 설정 → d |
| 텍스트 붙여넣기               | 일반 모드 → 붙여넣고 싶은 위치에 커서 위치 → p               |
| 파일 저장                     | 명령 모드(:) → w! + enter                                    |
| 파일 저장 + vim 종료          | 명령 모드(:) → wq! + enter                                   |
| vim 종료 (내용 저장되지 않음) | 명령 모드(:) → q! + enter                                    |


* 붙여넣기 계단 현상 발생 시 : `:set paste` 입력 후 복사
* [vim 단축키](https://string.tistory.com/51)