---
layout: post
title:  "Understanding Policy"
subtitle: "Deep RL Tutorial part-2"
date:   2018-01-10 20:15:00 +0530
comments: true
published: true
permalink : /deeprltutorial/part2/
categories: reinforcement-learning
description: "Understanding Policy is always central to understanding RL. In this section, we will give a glimpse of policy update in MDPs."
---


## Learning Policy

![Rat in Map](/assets/img/ratinmap.png)

$$ \pi (s) : S \rightarrow A $$

Our dear rat, is on a mission to discover most rewarding path to cheese. He can either climb up through the curtains and jump on kitchen floor, or he can climb via plants in balcony and enter kitchen from window. To learn the most optimal policy is rat's goal, and is also the goal of Reinforcement Learning. A policy charts out the action to be taken in a given state. Think of it like SIRI in navigation. At each turn(state), SIRI tells you the next best action.

In the next section,  we will learn bellman equations for  Value Iteration as a way of getting best policy.

## Bellman Equations  
*Bellman Optimality Equation for Value Iteration*


If you can see, the above thug life image of our rat, he is currently in state S1, and we need to find a way to update its state values.

$$ V^{(i+1)}(s_1) \leftarrow  \underbrace{max}_{s \in \{s_2,s_3\}} R_{a(s_1,s)} + \gamma  V^{(i)}(s) $$

$$ \pi^{(i+1)}(s_1) = \text{action}\in \{ a_1,a_2 \} \text{ with max future value } \{ V(s_1),V(s_2) \} $$

It means we change our policy greedily based on higher Value. Let us say $R_2+\gamma V(s_2)$ is 40, and $R_3+\gamma V(s_3)$ is 50, then we select action $a_3$ as it leads us to higher value. In short you update Values of states first and then update policy. You keep repeating these steps simultaneously(unlike policy iteration), until the value and policy converges. 

I know, math has started picking up pace, but there is one additional concept which needs to be brought. The outcome of action may be probabilistic. This means, our dear rat may jump from curtain to kitchen, but it can have a crash too. So, these cruel mathematicians added another layer of complexity, with outcome of action being probabilistic. This means same acton a, can lead you to different states probabilistically. For example there a long pit in front of you, the action is jumping, but there is 50-50 chances of either going across the pit, or landing in the pit with mud water. 

![Rat jumping from curtain](/assets/img/jumpingratmdpcropped.jpg)



| Bellman Expectation Equation | Bellman Optimality Equation | 
| ---------------------------- |-----------------------------| 
| Evaluating a given policy  | Involves learning an optimal policy |
| $$ \scriptsize v_{\pi}^{(i+1)}(s) \leftarrow E_{\pi}\{R_{next}+\gamma v_{\pi}(s_{next})\} $$ | $$ \scriptsize v^{(i+1)}(s) \leftarrow max\{ R_{next}+\gamma v(s_{next}) \} $$ |
| $$ \scriptsize v_{\pi}^{i+1}(s) \leftarrow E_{\pi}\{R_{sa}+\gamma \sum\limits_{s'\in S} p_{sas'}v_{\pi}(s')\} $$ | $$ \scriptsize v^{i+1}(s) \leftarrow max\{ R_{sa}+\gamma \sum\limits_{s'\in S} p_{sas'}v_{\pi}(s') \}  $$ |
|where $a = \pi(s)$ | where $a$ is action that offshoots from s.
|No policy is updated | Update $\pi(s)\leftarrow a$ with action ($a$) corresponding to maximum value in above stated value update.|  

Bellman equations give us a way of updating the value. They are not a set of equations which are restricted to only RL. They are typically associated with updates in DP (Dynamic Programming) problems. In value iteration, we simply select the policy  based on the action which higher expected reward from a given state. And we keep overwriting this expected reward to current value associated with state. This keeps going until we have reached equilibrium or a place where value is same as highest expected reward, and meanwhile the policy corresponding to this steady state is also optimal. *You can look up David Silver's notes to get proof of why convergence of value in Value Iteration also implies convergence of policy.*  


> **ALGORITHM-1 \| VALUE ITERATION**  
>   
> Initialize V(s) $\forall s \in S$  
> i = 0;  
> **while** values converge:  
> &nbsp;&nbsp;&nbsp;&nbsp; $$v^{(i+1)}(s) \leftarrow max \{ R_{sa}+\gamma \sum\limits_{s'\in S} p_{sas'}v_{\pi}(s') \} $$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *(Value Update)*  
> &nbsp;&nbsp;&nbsp;&nbsp; $$\pi (s) \leftarrow \underset{a}{argmax} \{ R_{sa}+\gamma \sum\limits_{s'\in S} p_{sas'}v_{\pi}(s') \} $$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *(Policy Update)*  
> **end while**  
>  
> *Note that here $R_{next}$ is reward for taking actions from current state. $s_{next}$ is one of the possible states an action can land it into* 
>

In the above equation, you can clearly observe that in each iteration policy and value update are happening. But there is a better way of doing it. What if we first fix the policy and let the values converge (lets say x number of iterations). Then used these converged values to update policy. Doesn't it seem better than constantly changing both policy and value together?

### Policy Iteration
Let's say you are in exam hall for derivation. There are 5 different ways(policy) of completing exam paper. Some people complete 1 markers first, some 5 markers first, etc. If you constantly keep changing policy in head, it is difficult to reach convergence/complete paper properly. Something similar happens in Policy Iteration. 


![ Policy Iteration ](/assets/img/policyiteration.png)

As shown in above illustration, policy iteration has two phases. Value Update and Policy Update. Previously in Value Iteration, we updated policy and value simultaneously. Policy Iteration is generally a better choice than Value Iteration. The pseudo-code of policy iteration is as given bellow.

The question comes up, can we learn action values instead of state values? The answer is yes. But it will converge with order of complexity $$ O(\#a^2.\#s^2)$$, which is slower as compared to 
$$ O(\#a.\#s^2)$$ in state value based Update.  
*Note that there are lot of advantages of action values, which will uncover when we discuss Model Free Methods*

> **ALGORITHM-2 \| POLICY ITERATION**  
>   
> Initialize V(s) $\forall s \in S$  
> i = 0;  
> **while** policy converges:  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*(policy same as before $\forall$ states)*  
> &nbsp;&nbsp;&nbsp;&nbsp; **while** values converge:  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $$v^{(i+1)}(s) \leftarrow max \{ R_{sa}+\gamma \sum\limits_{s'\in S} p_{sas'}v_{\pi}(s') \} $$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *(Value Update)*  
> &nbsp;&nbsp;&nbsp;&nbsp;**end while**  
> &nbsp;&nbsp;&nbsp;&nbsp; $$\pi (s) \leftarrow \underset{a}{argmax} \{ R_{sa}+\gamma \sum\limits_{s'\in S} p_{sas'}v_{\pi}(s') \} $$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *(Policy Update)*  
> **end while**  

Action values, also called q values are associated with state-action pair and not the action alone. The action is always with respect to some state. If you see the bellow unrolled illustration of value update, you can observe that maximization is happening at secondary nodes in case of action value update.

|  State Value Update | Action Value Update   |
|----------------------|-----------------------|
| ![State Value Update unrolled ](/assets/img/bellmanequationstatevaluefn.png) | ![Action Value Update unrolled](/assets/img/bellmanequationactionvaluefn.png) |
| $$ \scriptsize v^{i+1}(s) \leftarrow max \{R_{sa}+\gamma \sum\limits_{s'\in S} p_{sas'}v_{\pi}(s') \} $$ | $$ \scriptsize q^{i+1}(s,a) \leftarrow R_{sa}+\gamma \sum\limits_{s'\in S} p_{sas'}\underset{a'}{max} \{q_{\pi}(s',a') \} $$  |
| Update policy based on action with max expected reward or $$\scriptsize \underset{a}{argmax} \{ R_{sa}+\gamma \sum\limits_{s'\in S} p_{sas'}v_{\pi}(s') \}$$ | Update policy based on  action with higher $q(s,a)$ value |


For instance, $$ \small v_{s_1} = max\{R_{s,a_1}+ 0.5v_2+0.5v_3 ,R_{s,a_2}+ 0.5v_4+0.5v_5 \}  $$ for state value update  and   $$ \small q_1 = R_{s,a_1} +  0.5*max\{ q_2,q_3 \} + 0.5*max\{ q_4,q_5 \}  $$ for value update illustration shown above.


### Limitations of MDP
In many real time scenario, we don't know these transition probabilities. Especially in dynamic scenarios like traffic behavior, action's outcome is difficult to be modeled. So the question comes up, if we can do something better than traditional MDP, by dealing with model free approaches. What this means is that we refuse to assume anything about outcome of an action from a given state. The obvious question is that how do we learn in such scenarios? Our dear rat is getting bored by watching so many equations, so he decided to stop all this mathematics, and just goes to collect cheese the way he knowns. Of course, he has bruises and burns, but he keeps unrolling his episodes of cheese finding skills. We are going to use the same approach by dropping all this assumption business and directly learn from episodes of exploration(greedy or otherwise).

> Curtain(S) $\rightarrow$ jump(A) $\rightarrow$ knee-injury(R)  $\rightarrow$ ground(S) $\rightarrow$ run(A) $\rightarrow$ $\cdots$



