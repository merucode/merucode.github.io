---
layout: default
title: React Social Login
parent: React
grand_parent: Frontend
nav_order: 10
---

# React Social Login
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

### 

## STEP 0. Related Site

* [Google JavaScript API로 로그인 참조 ](https://developers.google.com/identity/gsi/web/reference/js-reference?hl=ko)

* [React 구글 로그인 구현](https://stack94.tistory.com/entry/React-%EA%B5%AC%EA%B8%80-%EB%A1%9C%EA%B7%B8%EC%9D%B8Google-Login-%EB%A6%AC%EC%95%A1%ED%8A%B8React%EC%97%90%EC%84%9C-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EC%9E%90)



<br>



##  STEP 1. Client ID 얻기



* 프로젝트 생성
* API 및 서비스 → Oauth 동의화면
  * User type : 외부
  * 그 외 최소 정보 입력
* API 및 서비스 → 사용자 인증 정보 → 사용자 인증 정보 만들기 → OAuth 클라이언트 ID

* Client ID 확인

  

<br>



## STEP 2. Google Login 구현

* `bash`

  ```bash
  $ npm install @react-oauth/google@latest
  ```

* `components/GoogleLoginButton.js`

  ```react
      import {GoogleLogin} from "@react-oauth/google";
      import {GoogleOAuthProvider} from "@react-oauth/google";
  
      const GoogleLoginButton = () => {
          const clientId = 'clientID'
          return (
              <>
                  <GoogleOAuthProvider clientId={clientId}>
                      <GoogleLogin
                          onSuccess={(res) => {
                              console.log(res);
                          }}
                          onFailure={(err) => {
                              console.log(err);
                          }}
                      />
                  </GoogleOAuthProvider>
              </>
          );
      };
  
      export default GoogleLoginButton
  ```

  



