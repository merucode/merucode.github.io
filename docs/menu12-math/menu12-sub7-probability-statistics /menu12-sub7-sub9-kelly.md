---
layout: default
title: Kelly
parent: Probability and Statistics
grand_parent: Math
nav_order: 1
---

# Kelly
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

## STEP 0. Reference Site

* [[youtube]12Math-인생에 꼭 필요한 수학 스킬](https://www.youtube.com/watch?v=C3Sdc_7e5Og)

## STEP 1. 이항분포와 정규분포

* 일반적인 확률분포
	* B(n, p) ~ N(np, np(1-p))
		* n: 횟수, p: 확률, np: 평균, np(1-p): 분산(σ2)
	* 표준편차 σ :(np(1-p)^1/2)
	* ±1.96σ(약 2σ) 표준편차 범위 내에 약 95% 분포

<br>

## STEP 2.  비율 계산

* 투자 비율 x(0 < x < 1)
	* 이익 후 자산: (1 - x) + ax
	* 손실 후 자산: (1 - x) + bx
* 예제 : 30%로 2배, 70%로 0.4배 일 때 적정 투자 비율
	* 이익 후 자산: 1 + 2x
	* 손실 후 자산: 1 - 0.6x
	* f(x) = (1+2x)<sup>3</sup>(1-0.6x)<sup>7</sup>
	* f'(x) = 6(1+2x)<sup>2</sup>(1-0.6x)<sup>7</sup> - 4.2(1+2x)<sup>3</sup>(1-0.6x)<sup>6</sup> = (6(1-0.6x) - 4.2(1+2x))(...) → x = 0.15
	* f(0.15) = 1.135 → 열번 투자 수행 시 기대 자산 증가률 1.135

* [그래프 계산기](https://www.desmos.com/calculator?lang=ko)

<br>

## STEP 3. 켈리 법칙

**f=p/a​−q/b**​
- f  : 베팅규모(보유자금 대비 베팅금액의 비율)
- a  : 순손해률(1원을 베팅하고 패배할 경우 순손해 a원)
- b  : 순이익률(1원을 베팅하고 승리할 경우 순이익 b원)
- p  : 승리 확률  
- q  : 패배확률(1−p)

