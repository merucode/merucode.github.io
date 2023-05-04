---
layout: default
title: React Summary
parent: React
grand_parent: Frontend
nav_order: 1
---

# React Summary
{: .no_toc}

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
<!------------------------------------ STEP ------------------------------------>

## STEP 1. React Basic

### Step 1-1. **`bash`**

```bash
$             # node.js 설치
$ node -v     # node.js 버전 확인 
$ npm -v      # npm 버전 확인

$ npm init react-app .    # 현재 폴더에서 react 프로젝트 시작
$ npm run start           # 개발서버 실행(localhost:3000)


```

### Step 1-2. **`.js`**

```javascript
// 기본 구성
import ReactDOM from 'react-dom/client';
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<h1>안녕 리액트!</h1>);   // render 안에 JSX

// fragment
root.render(<>
  <h1>안녕 리액트!</h1>
  <h1>안녕 리액트!</h1>
  </>); 

//  




//




//

```