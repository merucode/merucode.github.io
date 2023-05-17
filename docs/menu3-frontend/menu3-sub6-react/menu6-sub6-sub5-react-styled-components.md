---
layout: default
title: React Styled Components
parent: React
grand_parent: Frontend
nav_order: 5
---

# React Styled Components
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

## STEP 1. React Styled Components Basic

### Step 1-1. Pros

* CSS 클래스 이름이 겹치는 문제 해결(컴포넌트가 무수히 많은 경우)
* 재사용하는 CSS 코드를 관리하기 쉬움

### Step 1-2. Install

* `bash`(react 프로젝트 안)

  ```bash
  npm install styled-components
  ```

* 이번 수업은 Styled Components 버전 5 기준

### Step 1-3. Hello Styled

* `styled.tagname`의 `tagname` 부분에는 스타일을 적용할 HTML 태그 이름을 씁니다. 그리고, 바로 뒤에 템플릿 리터럴 문법으로 CSS 코드를 적습니다.

```
/* Button.js */
import styled from 'styled-components';

const Button = styled.button`
  background-color: #6750a4;
  border: none;
  color: #ffffff;
  padding: 16px;
`;

export default Button;


/* App.js */
import Button from './Button';

function App() {
  return (
    <div>
      <Button>Hello Styled!</Button>
    </div>
  );
}

export default App;
```

### Step 1-4. Nesting

* Nesting은 CSS 규칙 안에서 CSS 규칙을 만드는 걸 말하는데요. Nesting을 활용하는 두 가지 방법인 `&` 선택자와 컴포넌트 선택자에 대해 알아보겠습니다.

* `&` 선택자 : `&`는 부모 선택자 의미(&:hover = .Button:hover)

  ```react
  // 버튼 컴포넌트를 호버하거나 클릭했을 때 배경색 변경
  const Button = styled.button`
    background-color: #6750a4;
    border: none;
    color: #ffffff;
    padding: 16px;

    &:hover,
    &:active {
      background-color: #463770;
    }
  `;

  export default Button;
  ```

* 컴포넌트 선택자 : ${Icon}같이 컴포넌트 자체를 템플릿 리터럴 안에 입력
  * `&`와 자손 결합자를 사용하는 경우에는 &를 생략할 수 있습니다. 즉 `${Icon}`만 써도 똑같이 동작(권장)(`& ${Icon}` = `${Icon}` = `.StyledButton .Icon`)

  ```react
  // 버튼 안에 아이콘 배치
  import styled from 'styled-components';
  import nailImg from './nail.png';

  const Icon = styled.img`
    width: 16px;
    height: 16px;
  `;

  const StyledButton = styled.button`
    background-color: #6750a4;
    border: none;
    color: #ffffff;
    padding: 16px;

    & ${Icon} {
      margin-right: 4px;
    }

    &:hover,
    &:active {
      background-color: #463770;
    }
  `;

  function Button({ children, ...buttonProps }) {
    return (
      <StyledButton {...buttonProps}>
        <Icon src={nailImg} alt="nail icon" />
        {children}
      </StyledButton>
    );
  }

  export default Button;
  ```

* Nesting은 여러 겹 사용 가능

  ```react
  const StyledButton = styled.button`
    ...
    &:hover,
    &:active {
      background-color: #7760b4;

      ${Icon} {
        opacity: 0.2;
      }
    }
  `;
  ```

### Step 1-5. Dynamic Styling

```react
/* App.js */
function App() {
  return (
    ...
    <Button size="small">small</Button>
    <Button size="large" round>round large</Button>
    ...
  )
}
```

* ${ ... } 안에 값(변수) 사용하기

  ```react
  const SIZES = {
    large: 24,
    medium: 20,
    small: 16
  };

  const Button = styled.button`
    ...
    font-size: ${SIZES['medium']}px;
  `;
  ```

* ${ ... } 안에 `pros` 또는 함수 사용하기 : 템플릿 리터럴의 기능이 아니라 Styled Components가 내부적으로 처리

  ```react
  const SIZES = {
    large: 24,
    medium: 20,
    small: 16
  };

  const Button = styled.button`
    ...
    font-size: ${(props) => SIZES[props.size]}px;
  `;
  ```

  ```react
  // 구조 분해(Destructuring)
  font-size: ${({ size }) => SIZES[size]}px;
  font-size: ${({ size }) => SIZES[size] ?? SIZES['medium']}px;
  // 널 병합 연선자 권장(size props가 undefined 경우 처리 가능)  
  ```

* 논리 연산자 사용하기

  ```react
  const Button = styled.button`
    ...
    ${({ round }) => round && `
        border-radius: 9999px;
      `}
  `;

  // 삼항 연산자
  border-radius: ${({ round }) => round ? `9999px` : `3px`};
  ```

### Step 1-6. Style Inheritance

* `styled()` 함수로 상속하기

  ```react
  // Button style 상속
  // 몇 가지 스타일만 추가하거나 덮어써서 원하는 컴포넌트 적성
  import styled from 'styled-components';
  import Button from './Button';

  const SubmitButton = styled(Button)`  
  ...
  `;
  ```


* JSX로 직접 만든 컴포넌트에 styled() 사용
  * Styled Components를 사용하지 않고 직접 만든 컴포넌트에는 클래스 이름을 내려준 후에 styled() 함수로 상속

  ```react
  /* App.js */
  import styled from 'styled-components';
  import Button from './Button';
  import TermsOfService from './TermsOfService';

  const StyledTermsOfService = styled(TermsOfService)`
    ...
  `; 

  function App() {
  return (
    <div>
      <StyledTermsOfService />
      <SubmitButton>계속하기</SubmitButton>
    </div>
  );
  }

  export default App;
  `;


  /* TermsOfService.js */
  function TermsOfService({ className }) {
    return (
      <div className={className}>
        ...
      </div>
    );
  }
  ```

* `css` function : CSS 코드들을 변수처럼 저장해서 여러 번 다시 사용하고 싶을 때

  ```react
  import styled, { css } from 'styled-components';

  const SIZES = {
    large: 24,
    medium: 20,
    small: 16
  };

  const fontSize = css`
    font-size: ${({ size }) => SIZES[size] ?? SIZES['medium']}px;
  `;

  const Button = styled.button`
    ...
    ${fontSize}
  `;

  const Input = styled.input`
    ...
    ${fontSize}
  `;
  ```

  ```react
  // 함수를 삽입하지 않는 단순한 문자열이라면 일반적인 템플릿 리터럴
  const boxShadow = `
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
  `;

  // 하지만, 이런 경우에도 항상 css 함수를 사용하도록 습관화하는 걸 권장
  const boxShadow = css`
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
  `;
  ```

