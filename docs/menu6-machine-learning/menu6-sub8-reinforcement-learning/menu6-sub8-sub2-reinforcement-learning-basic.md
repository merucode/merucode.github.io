---
layout: default
title: Reinforcement Learning Basic
parent: Reinforcement Learning
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

  *  Reason that is not max operator in front of q(s,a) equation
    *   우리가 평가하려는 a에 대한 value에서 a는 항상 최선의 행동을 의미하지는 않기 때문에 max 연산자가 앞에 붙지 않음
  *  𝝅(a`|`s) → max<sub>a</sub>
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



### Step 3-2. Prediction : Iterative Policy Evaluation(𝝅 given)

* **Method**

  1. Initialize table

  2. Update one state

     <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531023832922.png" alt="image-20230531023832922" style="zoom:67%;" />

     * 무의미한 값에 실제 값이 섞여 반복에 의해 실제 값에 가까워 짐

     * ex) 𝝅(동서남북`|`s) = 0.25, r = -1, P = 1, 초기 value = 0

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

## STEP 4. Don't Know MDP, and Small Problem : Prediction

### Step 4-1. Preview

* **Prediction**

  1. MonteCarlo Method
  2. Temporal difference

* **model(model of enviromnet)** : 액션에 대하여 환경이 어떻게 응답할지 예측하는 모든 것

* using **tabular method**(Example)

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531030203345.png" alt="image-20230531030203345" style="zoom:50%;" />

  * We know r=-1, P=1, but **system don't know about r, P**



### Step 4-2. MC(MonteCarlo) Learning

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

  * Above method is need **many episode to update v<sub>𝝅</sub>**(Because it need average)
    * Below expression is need only **one episode finishing to update  v<sub>𝝅</sub>**
    * Don't need to save N(s<sub>t</sub>)

  
  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531031043039.png" alt="image-20230531031043039" style="zoom:67%;" />
  
  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230531031114200.png" alt="image-20230531031114200" style="zoom:67%;" />
  



### Step 4-3. Implement MonteCalro Learning

* 4 things needed for implement 
  1. environment
  2. agent
  3. experience part
  4. learning part
* [code url](https://github.com/merucode/study_ML/blob/master/RL/basic/ch4_MCLearning.ipynb)



### Step 4-4. TD(Temporal Difference) Learning

* **MC and TD**

  | Items        | MC                                                           | TD                                                           |
  | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | Update point | One episode finish                                           | **After operate step**<br>Don't need to finish episode       |
  | Theory       | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602021630218.png" alt="image-20230602021630218" style="zoom:67%;" /> | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602021653149.png" alt="image-20230602021653149" style="zoom:67%;" /> |
  | Note         | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602021714113.png" alt="image-20230602021714113" style="zoom:67%;" /> |                                                              |

* **TD target**
  * r<sub>t+1</sub> +γ v<sub>𝝅</sub>(s<sub>t+1</sub>) 을 여러번 sampling 하여 평균을 내면 v<sub>𝝅</sub>(s<sub>t</sub>) 에 수렴
  * 즉, **r<sub>t+1</sub> +γ v<sub>𝝅</sub>(s<sub>t+1</sub>)** 는 우리의 목표(정답)가 되는 값이기 때문에 **TD target**

* **TD Learning Algorithm**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602021856194.png" alt="image-20230602021856194" style="zoom:67%;" />

  * ex> s<sub>0</sub> → s<sub>1</sub> →  s<sub>2</sub> →  ...  s<sub>11</sub> →  종료

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602021927693.png" alt="image-20230602021927693" style="zoom:50%;" />



### Step 4-5. Implement TD Leaning

* [code url](https://github.com/merucode/study_ML/blob/master/RL/basic/ch4_TDLearning.ipynb)



### Step 4-6. MC vs TD

* **Compare MC and TD**

  | Items    | MC                                                           | TD                                                           |
  | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | Leaning  | Episodic MDP<br>(종료상태가 있는 것)                         | Episodic MDP<br>Non-Episodic MDP                             |
  | Bias     | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602021630218.png" alt="image-20230602021630218" style="zoom:67%;" /><br>**unbiased** | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602021653149.png" alt="image-20230602021653149" style="zoom:67%;" /><br>**biased** |
  | Variance | 서울시청 → 강릉<br>변동성 큼(작은 **α**)                     | 서울시청 → 앞 편의점<br>변동성 낮음(큰 **α**)                |

* **Reason that TD is biased**

  * **r<sub>t+1</sub> +γ v<sub>𝝅</sub>(s<sub>t+1</sub>)**(실제 TD target) is **unbiased**

  * **r<sub>t+1</sub> +γ V(s<sub>t+1</sub>)**(우리가 사용하는 TD target) is **baised**

    * TD에서는 실제 값(**v<sub>𝝅</sub>**)모름으로 추측 값(**V**)을 정답으로 사용하여 update 수행

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602024721754.png" alt="image-20230602024721754" style="zoom: 67%;" />

​	

### Step 4-7. n Step MDP

* **Relation between MC and TD**

  * N step TD target

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602024943233.png" alt="image-20230602024943233" style="zoom:67%;" />

  * N = T(end point) → MC

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602025105313.png" alt="image-20230602025105313" style="zoom: 67%;" />



<br>



<!------------------------------------ STEP ------------------------------------>

### 초안 작성

## STEP 5. Don't Know MDP, and Small Problem : Control

### Step 5-1. Preview

* **Control**
  1. MC Control
  2. TD Control : SARSA
  3. Q Leaning



### Step 5-2. MC Control

* We want to use **Policy interation** but **don't know MDP(r<sub>S</sub><sup>a</sup>, P<sub>ss'</sub><sup>a</sup>)**

  * don't know **r<sub>S</sub><sup>a</sup>**  → don't select 𝝅<sub>greedy</sub>
  * don't know **P<sub>ss'</sub><sup>a</sup>** → don't select action(don't know result of action)

* **Solution**

  1. Policy evaluation : MC Learning

  2. Policy improvement : use Q instead of V

     | V                                                            | Q                                                            |
     | ------------------------------------------------------------ | ------------------------------------------------------------ |
     | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602032625474.png" alt="image-20230602032625474" style="zoom:67%;" /> | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602032545261.png" alt="image-20230602032545261" style="zoom:67%;" /> |
     |                                                              | Need q(s,a) value to use MC                                  |

  3. exploration

     * 무조건 한 방향만 Action 시 더 좋은 value를 놓칠 수 있음

     * Introduce **ε-greed**(decay)

       <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602032851236.png" alt="image-20230602032851236" style="zoom:80%;" />

       * decay : 초기 높은 ε, 후기 낮은 일정 ε

* **MC Control**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602032958062.png" alt="image-20230602032958062" style="zoom:67%;" />



### Step 5-3. Implement MC Control

* **Example**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602033510256.png" alt="image-20230602033510256" style="zoom:67%;" />

* [code url](https://github.com/merucode/study_ML/blob/master/RL/basic/ch5_MCContorl.ipynb)



### Step 5-4. TD Control : SARSA

* **Policy evaluation : using TD instead of MD**

  * It's help to **update not end of episode but end of step**

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602034929327.png" alt="image-20230602034929327" style="zoom:67%;" />

* **SARSA**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602035015216.png" alt="image-20230602035015216" style="zoom:67%;" />

  * **TD Target**

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602035151685.png" alt="image-20230602035151685" style="zoom:80%;" />

    * **기대값 안의 수식 Sampling → 실제 기대값에 가까워짐**

  * **SARSA**

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602035213985.png" alt="image-20230602035213985" style="zoom:80%;" />



### Step 5-5. Implement TD Contorl : SARSA

* [code_url](https://github.com/merucode/study_ML/blob/master/RL/basic/ch5_SARSA.ipynb)



### Step 5-6. TD Contorl : Q Learning

* **On/Off Policy**
  * on-policy : same **target policy** and **behavior policy**
  * off-policy : different **target policy** and **behavior policy**

* **Target/Behavior Policy**

  * target policy : 강화하고자 하는 목표가 되는 정책
  * behavior policy : 실제 환경과 상호작용하여 경험을 쌓는 정책

* **Off Policy Adventage**

  1. reuse past experience
  2. learning from data of person
     * system learn from (s, a, r, s') data
  3. 1:N or N:1 learning is possible

* **Theory Background**

  - From bellman optimality equation

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602040912800.png" alt="image-20230602040912800" style="zoom: 67%;" />

    - If we know **q<sub>*</sub>**, optimal policy is below(just move to highest **q<sub>*</sub> action**)

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602041116527.png" alt="image-20230602041116527" style="zoom:67%;" />

    * **Q에 대하여 greed policy**

  - So, our purpose is getting **q<sub>*</sub>**

    * From bellman optimality equation(step 0)

      <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602041418605.png" alt="image-20230602041418605" style="zoom:67%;" />

    * Replace E with sampling value

* **Compare SARSA and Q-Learning**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602041624145.png" alt="image-20230602041624145" style="zoom:67%;" />

  | Items           | SARSA                                                        | Q-Learning                                                   |
  | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | Behavior Policy | Q에 대해<br>ε-greedy                                         | Q에 대해<br/>ε-greedy                                        |
  | Target Policy   | Q에 대해<br/>ε-greedy                                        | Q에 대해<br/>greedy                                          |
  | Policy          | on                                                           | off<br>Difference Behavior and Target policy                 |
  | Theory          | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602042239508.png" alt="image-20230602042239508" style="zoom:67%;" /><br>Bellman Expectaion | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602042251835.png" alt="image-20230602042251835" style="zoom:67%;" /><Br>Bellman Optimality |
  | Difference      | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602042328635.png" alt="image-20230602042328635" style="zoom:80%;" /> | <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602042335110.png" alt="image-20230602042335110" style="zoom:80%;" /><br>𝝅 only greedy select for Q |
  | Note            |                                                              | 𝝅<sub>*</sub> is dependent on environment(q)                 |



### Step 5-6. Implement Q-Learning

* [code url](https://github.com/merucode/study_ML/blob/master/RL/basic/ch5_QLearning.ipynb)



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP 6. Deep RL

### Step 6-1. Function to save data

* To solve large scale problem 
  * We use deep RL(Deep Learning + Reinforce Learning)
* Don't use tabular method to solve problem have to many state(바둑, continuous state space problem ...)
  * Introduce function to save data
* **Function Generalization**
  * **Make general function for state/action data express using parameters(w or theta)**
  * Small storage to save results of learning
  * Use deep learning to find general function



### Step 6-2. Implement Function Generalization

* **Example**

  * get data from below function and make general function

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602044713357.png" alt="image-20230602044713357" style="zoom:67%;" />

* [code url](https://github.com/merucode/study_ML/blob/master/RL/basic/ch6_Fitting.ipynb)


<br>

##  여기부터 빠르게 진행 복습 필요

<!------------------------------------ STEP ------------------------------------>

## STEP 7. Model free, Large state/action space

### Step 7-1 Preview

* **Deep ML**

  1. Value based : v<sub>𝝅</sub>(s), q<sub>𝝅</sub>(s,a) → neural net
  2. Policy Based : 𝝅(a`|`s) → neural net

* **RL Agent**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602050605538.png" alt="image-20230602050605538" style="zoom:67%;" />

  1. value based
     * Select action from value(q(s,a))
     * SARSA, Q-Learning
     * When select action, **only greedy select high q(s,a)**
       * So, don't need 𝝅(a`|`s) and **q(s,a)** is rule of 𝝅
  2. policy based
     * Select action from 𝝅(a`|`s)
     * When select action, **only use 𝝅**. So, don't need value, evaluation function
  3.  Actor-Critic
     * Select action from both value and policy
       * Actor : 𝝅
       * Critic : v(s) or q(s,a)

* **value based**
  * 𝝅가 고정되었을 떄, 𝝅의 가치함수 v<sub>𝝅</sub>(s)를 학습
  * value network
    * θ is neural net parameters
    * purpose : learning proper θ, to v<sub>θ</sub>(s) get proper value per states



## Step 7-2. Learn Value Network

* Value network of v<sub>θ</sub>(s)(𝝅 고정)

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602051537290.png" alt="image-20230602051537290" style="zoom:50%;" />

* **Loss function**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602051650244.png" alt="image-20230602051650244" style="zoom:67%;" />

  * 위 식은, 어떤 s에 대한 것인지가 없음

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602051750947.png" alt="image-20230602051750947" style="zoom:67%;" />

  * 𝝅에 의한 sampling 통해 기대값을 근사적으로 계산 가능, 

  * gradient 수행하면(상수 생략)

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602051915991.png" alt="image-20230602051915991" style="zoom:67%;" />

  * 𝝅에 의해 상태 s에 들어갈 경우

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602052018759.png" alt="image-20230602052018759" style="zoom:67%;" />

  * 위 식 하나로만은 성립하지 않으나, Sampling을 통해 수많은 값으로 우변 평균을 내면 좌변에 근사하게 됨

  * θ Update

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602052149815.png" alt="image-20230602052149815" style="zoom:67%;" />

  * Sampling 및 gradient descent 의해 v<sub>θ</sub>(s)는 v<sub>true</sub>(s)에 근사하게 됨

* **But, we don't know  v<sub>true</sub>(s)**

  * To get  v<sub>true</sub>(s)
    1. MC return
    2. TD target



### Step 7-3 MC Return

* **MC return**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602052521771.png" alt="image-20230602052521771" style="zoom:67%;" />

  * G<sub>t</sub> 사용 가능한 이유는 실제 가치함수 정의가 G<sub>t</sub>의 기대값이기 때문에(G<sub>t</sub> = v<sub>ture</sub>(s<sub>t</sub>))

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602052542477.png" alt="image-20230602052542477" style="zoom:67%;" />

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602052554342.png" alt="image-20230602052554342" style="zoom:67%;" />



### Step 7-4. TD Target

* Introduce TD target instead of G<sub>t</sub>

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602052845953.png" alt="image-20230602052845953" style="zoom:67%;" />

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602052857615.png" alt="image-20230602052857615" style="zoom:67%;" />

* TD target is 상수 not function of θ → θ 편미분시 TD target 항 0
  * 만약 상수 취급 안한다면 TD target(추측 정답값)도 변하게 되어 모델 안전성이 떨어짐



### Step 7-5. Deep Q learning

* Value based agent don't have **explicit policy(𝝅)**
  * 𝝅 is not exist
  * act for greedy at q(s,a) → **implicit policy q(s,a)**

* **Theory Background**

  * From Bellman optimality Equation

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602053402193.png" alt="image-20230602053402193" style="zoom:67%;" />

    * 추측 정답인 TD target 과 추측인 Q(s,a) 사이 차이를 줄이는 방향으로 업데이트

  * Loss function

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602053505409.png" alt="image-20230602053505409" style="zoom:67%;" />

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602053534497.png" alt="image-20230602053534497" style="zoom:67%;" />

    * 정답 Q<sub>*</sub>과 Q<sub>θ</sub> 차이를 줄이는 방향으로 업데이트

  * use mini-batch instead of E

* **Deep Q Learning : pseudo code**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602053818494.png" alt="image-20230602053818494" style="zoom:67%;" />

  * 3-A real act(Behavior policy : eps-greedy)

  * 3-C is not real action but operate to calculate TD value(Target policy : greedy)

  * off-policy

  * When Implement we just define L(θ). don't need gradient of L(θ)

    * optimizer and 역전파가 알아서 계산 수행해줌

      <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602054440045.png" alt="image-20230602054440045" style="zoom:67%;" />



### Step 7-6. Implement DQN

* **Experience Replay**

  * episode is consist of many transitions(one transition = e<sub>t</sub>)

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602054718490.png" alt="image-20230602054718490" style="zoom:67%;" />

  * use replay buffer to **reuse transition**

    * If save new data, delete oldest data

    * mini-batch extract data from replay buffer

      * mini-batch is consist of not continuous data. It's mean they are small correalation

        → help to improve performance

  * only use to off-policy algorithm

* **Target Network**

  <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602055229039.png" alt="image-20230602055229039" style="zoom:80%;" />

  * Q-Learning 정답

    <img src="./../../../images/menu6-sub8-sub2-reinforcement-learning-basic/image-20230602055312486.png" alt="image-20230602055312486" style="zoom:80%;" />

    * θ에 대한 함수이므로 θ가 변하면 변하게 됨
    * 일정 시간 freezing 시키는 θ<sub>i</sub><sup>-</sup> 도입

  * **Target and Q network**

    * Target network : Calculate answer, freezing θ<sub>i</sub><sup>-</sup>
    * Q network :  Learning, θ updated, 일정 주기마다 θ → θ<sub>i</sub><sup>-</sup>


* [code url](https://github.com/merucode/study_ML/blob/master/RL/basic/ch7_DQN.ipynb)


<br>



<!------------------------------------ STEP ------------------------------------>



https://github.com/seungeunrho/RLfrombasics/blob/master/ch8_DQN.py

<br>



<!------------------------------------ STEP ------------------------------------>





<br>



<!------------------------------------ STEP ------------------------------------>





