---
layout: default
title: Reinforcement Learning Basic
parent: Reinforcement Leaning
grand_parent: Machine Leaning
nav_order: 2
---

# Reinforcement Learning Basic
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

## STEP 1. MDP(Markov Decision Process)

### Step 1-1. MDP

* **MDP**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531014605568.png" alt="image-20230531014605568" style="zoom:80%;" />

* **Component**

  | Items | Description                   | Equation                                                     |
  | ----- | ----------------------------- | ------------------------------------------------------------ |
  | S     | Group of states               |                                                              |
  | A     | Group of actions              |                                                              |
  | P     | Transition probability matrix | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531015115476.png" alt="image-20230531015115476" style="zoom: 67%;" /> |
  | R     | Reward function               | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531015131196.png" alt="image-20230531015131196" style="zoom:67%;" /> |
  | γ     | Damping factor                |                                                              |



### Step 1-2. Policy function and two value function

* **Policy function**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531015546314.png" alt="image-20230531015546314" style="zoom: 67%;" />

  * Policy function is related with agent(not environment)

* **If 𝝅 is given**, we can get **two value function**(two value function is depend on 𝝅)

* **State value function**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531015856727.png" alt="image-20230531015856727" style="zoom: 67%;" /> 

* **State-action value function**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531020005369.png" alt="image-20230531020005369" style="zoom:67%;" />



### Step 1-3. Prediction, Control

* **Prediction** : evaluate value of state with given 𝝅
* **Control** : find best policy function(𝝅)



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 2. Bellman Equation

### Step 2-1. Bellman Expectation Equation

* **Bellman Expectation Equation**(𝝅 is given)

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531020529338.png" alt="image-20230531020529338" style="zoom:67%;" />

  * **Step 1 Example**

    | v                                                            | q                                                            |
    | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531021438490.png" alt="image-20230531021438490" style="zoom: 67%;" /> | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531021557094.png" alt="image-20230531021557094" style="zoom:67%;" /> |
    | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531021512841.png" alt="image-20230531021512841" style="zoom:67%;" /> | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531021753802.png" alt="image-20230531021753802" style="zoom:67%;" /> |

    

  * What it means to **know MDP** is **knowing r<sub>S</sub><sup>a</sup>, P<sub>ss'</sub><sup>a</sup>**
    * **Know MDP** →  Step 2. equation → Model-based, planning 
    * **Don't know MDP** → Step 0. equation → Model-free, sampling

### Step 2-2. Optimal Value/Policy

* **Optimal Value/Policy**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531022003525.png" alt="image-20230531022003525" style="zoom: 80%;" />

  * 상태 별(s, s' ...)로 가장 높은  value의 정책의 다른 경우

    → 각각의 정책(𝝅<sub>s</sub>, 𝝅<sub>s'</sub> ...)을 조합해 새로운 정책(𝝅<sub>*</sub>) 가능

  * MDP는 𝝅<sub>*</sub>가 반드시 존재함

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531022311783.png" alt="image-20230531022311783" style="zoom:67%;" />



### Step 2-3. Bellman **Optimality** Equation

* **Bellman Optimality Equation**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531022427461.png" alt="image-20230531022427461" style="zoom:67%;" />

  *  𝝅(a|s) → max<sub>a</sub>
    * **Bellman Expectation Equation** : 2 stochastic factor(P, 𝝅)
      * use to evaluate 𝝅
    * **Bellman Optimality Equation** : 1 stochastic factor(P)
      * use to get best value

<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 3. Know MDP, and Small Problem

### Step 3-1. Preview

* **Prediction : Iterative policy evaluation**

* **Control**

  1. **Policy iteration**
  2. **Value iteration**

* using **tabular method**(Example)

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531023414340.png" alt="image-20230531023414340" style="zoom:67%;" />



### Step 3-2. Prediction : Iterative Policy Evaluation

* **Method**

  1. Initialize table

  2. Update on state

     <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531023832922.png" alt="image-20230531023832922" style="zoom:67%;" />

     * 무의미한 값에 실제 값이 섞여 반복에 의해 실제 값에 가까워 짐

     * ex) 𝝅(동서남북|s) = 0.25, r = -1, P = 1, 초기 value = 0

       <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531023934269.png" alt="image-20230531023934269" style="zoom:50%;" />

  3. apply 2. on all states

  4. iterate 2.~3.

     <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531024132739.png" alt="image-20230531024132739" style="zoom:50%;" />



### Step 3-3. Contrl : Policy iteration

* **정책 평가/정책 개선 반복**

  * Policy evaluation : iterative policy evaluation
  * Policy improvement : get 𝝅<sub>greedy</sub> from policy evaluation

* **Greedy Policy**

  * Act to get more good value
  *  𝝅<sub>greedy</sub> is improved over 𝝅

* **early stopping**

  * There is no need to repeat to the limit, because even a **little repetition is worthwhile**

* **Method**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531024636710.png" alt="image-20230531024636710" style="zoom: 67%;" />



### Step 3-4. Control : Value iteration

* Get **optimal policy** from **optimal value** derived from **bellman optimality equation**

* **Method**

  1. Get optimal value From bellman optimality equation using iterative policy evaluation

     <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531025334910.png" alt="image-20230531025334910" style="zoom:67%;" />

     * ex)

       <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531025416567.png" alt="image-20230531025416567" style="zoom:50%;" />

  2. Get 𝝅<sub>*</sub> from optimal value

     * 𝝅<sub>*</sub> is greedy policy to optimal value

<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 4. Don't Know MDP, and Small Problem

### Step 4-1. Preview

* **Prediction**

  1. MonteCarlo Method
  2. Temporal difference

* **Control**

* **model(model of enviromnet)** : 액션에 대하여 환경이 어떻게 응답할지 예측하는 모든 것

* using **tabular method**(Example)

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531030203345.png" alt="image-20230531030203345" style="zoom:50%;" />

  * We know r=-1, P=1, but system don't know about r, P



### Step 4-2. MonteCarlo Method

* Get value from **many sampling**

* **Method**

  1. Initialize table : (0, 0)

     * N(s) : counts of enter, V(s) : sum of returns

  2. Experience : arrived at S<sub>T</sub>

     <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531030039520.png" alt="image-20230531030039520" style="zoom:50%;" />

  3. Update table

     * N(s<sub>t</sub>) ← N(s<sub>t</sub>) + 1

     * V(s<sub>t</sub>) ← V(s<sub>t</sub>) + G<sub>t</sub>

       <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531030753721.png" alt="image-20230531030753721" style="zoom:50%;" />

  4. Calculate value

     <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531030900628.png" alt="image-20230531030900628" style="zoom:67%;" />

* **Version of partial update**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531031043039.png" alt="image-20230531031043039" style="zoom:67%;" />

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531031114200.png" alt="image-20230531031114200" style="zoom:67%;" />

  * Don't need to save N(s<sub>t</sub>)



### Step 4-3. Implement MonteCalro Method

* 4 things needed for implement 
  1. environment
  2. agent
  3. experience part
  4. learning part
* [code url](https://github.com/merucode/study_ml/blob/master/reinforcement/basic/ch1_montecarlo/Untitled.ipynb)



