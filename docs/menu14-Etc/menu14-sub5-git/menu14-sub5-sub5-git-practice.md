---
layout: default
title: Git Practice
parent: Git
grand_parent: Etc
nav_order: 2
---

# Git Practice

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



## STEP 1. Local Git Delete

```bash
### 로컬 저장소의 git 히스토리 삭제
$ rm -rf .git	

### git 초기화
$ git init		

### project commit
$ git add . 	
$ git commit -m "first project commit"

### push
$ git remote add origin {github repository url}
$ git push -u origin master
```

