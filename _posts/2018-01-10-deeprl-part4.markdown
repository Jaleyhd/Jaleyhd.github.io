---
layout: post
title:  "Reinforcement Learing "
subtitle: "Deep RL Blog Series part-1"
date:   2018-01-09 20:15:00 +0530
comments: true
published: false
permalink : /deeprltutorial/part2/
categories: reinforcement-learning
description: "In this blog, we talk about creating ground Zero for understanding RL. We intend to use his as a base to learn more RL Techniques in the subsequent blogs where we will be doing cool stuff like training Atari Games"
---

## Why is RL a Game Changer?
When you start binge watching David Silver RL Lectures on a bright Saturday Morning, its hard to not think that there is a whole world out there yet to be explored. Reinforcement Learning is not a new concept. In 1950's Richard Bellman was working hard to explain behavior of time-dynamic systems. He became father of what can be considered a remarkable problem-solving techinique, which is quite popular amongst CS-Algo students. Its **Dynamic Programming**. It is most clever way of reusing computations of optimal substructures, for example in Knapsack Problem. 

In Reinforcement Learning, the substructures are subsolutions. 
It is  The idea of optimality in Markovian processes, specifically Markov Decision processes can be said to foundation to RL Techniques. This contribution by Bellman, in 1957, also known as Bellman Optimality Equation, can explain how cooking pans arrive at steady state temperature after 10 minutes. 

> If you have mercury thermometers at home for measuring fever, you can observe, how mercury comes to steady state. This can be looked as markov process with single state. However, things can be pretty messed up, in case of pan, where different parts of pan(continuous state) are at different temperature, yet at an equilibrium. To explain such behavior in thermodyamics, we can use Bellman Equations.  

The textbook examples of mouse discovering cheese may seem oversimplified, but it is not. It provides all we need, to define the distressing jargons of RL.

How many of you readers have watched Ratatouile Movie? If you have not, do watch it first, before you read further, because  we will be using life of Chef Remy(the great rat) in the blog to explain Reinforcement Learning is from that movie only. It is strictly non-negotiable prerequisite :p. 


Remy the rat is the best example of RL Agent, trying to discover optimal way of getting maximum reward (cheese) in streets of paris (RL Environment). Agent interacts with environment and learns how to act better. 


![RL Jargons](/assets/img/rljargons.png)


***Observation*** $(O_t)$ : It is how the agent *currently* perceives the environment. It can be image, sound, etc.

***State*** $(S_t)$ : David Silver describes Agent State as agent's internal representation of environment. It can be  any function of observation history. Your current position with respect to starting point is one kind of state. Latitude, Longitude is another example of state variable.

> State is different from observation, because it has more summarized and tangible information. There is an environment state (how environment perceives itself) and an Agent state. Here, we are referring to Agent State

**Action** $(A_t)$ : Agent's choices in a given state. For example taking left turn, taking right turn, applying break, etc can be considered actions while driving a car. 


**Reward** $(R_t)$ : Reward is the incentive given by environment to agent, for taking an action at a given state.  

> Reward is associated with action (at a given state) and not with the state itself. This is a very common conceptual mistake amongst the beginners. 


**Policy** $\pi(S_t)$ : Agent's model for deciding action in a given state. For example uniform random policy means, take any action randomly in a given state. Absolute greedy policy means, just go by immediate reward. For rat, it means, just sniff the smell of cheese, whichever direction has most aroma, go there (doesn't matter if there is rat-trap or boiling water in front of you). 


## No Brainer Policy
![Immediate Reward](/assets/img/immediatereward.png)

What is the simplest policy can follow in any environment? Well, we can just look at immediate reward and decide our next state. As you can see in the above diagram, such policy would take the yellow path, as it goes by immediate reward. So I guess you know what's the problem. The problem is, that this no-brainer policy doesn't know if there is another path which could eventually make it hit the jackpot. Thats the exact reason we need a dynamic model which takes into account future rewards. We will come back to this problem after a short detour of Markov Chain which is necessary to understand dynamic systems.

## Markov Chain
![Cupcake Markov Chain](/assets/img/cupcakemarkovchain.png)

In 2 broke girls, do you know why the cupcake business kick-started by X and Y did not take off? Well, its because they didn't meet pal of Z from Russia. Ya, you guessed it, Its Markov. Had they met Markov, he would have explained how customers are eating cupcakes. 

Now strawberry cupcake tastes great, but you are unlikely to get addicted to it, on the other hand, chocolate cupcakes, just hit the right spot in brain, releasing all endorphins. Therefore, you will keep buying chocolate cupcakes again and again.

Our target here is to find probability of buying strawberry cupcake opposed to buying chocolate cupcake. And so, we setout a simple experiment. With help of Markov, we track a customer's cupcake pattern and find out the transition probabilities based on counts. Let me explain. 


> #### Constructing Markov Transition model from scratch
> If a customer bought 101 cupcakes in 101 days. He transitioned from chocolate to strawberry only 5 times, kept holding on to chocolate for 45 times. Similarly he transitioned from strawberry to chocolate 25 times, and strawberry to strawberry another 25 times.
>
 $$ \small P(S | C) = P(C\rightarrow S) = \cfrac{\#C\rightarrow S}{\#C} = \cfrac{\#C\rightarrow S}{\#C\rightarrow S + \#C\rightarrow C} = \cfrac{5}{5+45} =0.1 $$
> 
>  Even though the transition model looks rock solid, there is fundamental assumption we have made. Any guesses? 
>
> **Markovian Property**  
> We assume that the customer behavior is markovian, which means, kind of cupcake I buy today can be totally determined by cupcake I bought yesterday. This means, in a markov chain, aka cupcake chain (unrolled in time), what you buy 5 days back becomes irrelevant if I know what you bought yesterday. Mathematicians state this as conditional independence. Formally , we can say that, future states becomes conditionally independent of past states, if we know current state. 

There are two ways of finding out probability of states from state transition model. First one is matrix way. Equate weighted sum of incoming transitions to state probability for each of the state. You get two equations, two variables and solution. 

$$ \begin{equation}0.9C+0.1S=C\end{equation} $$

$$ \begin{equation}0.5C+0.5S=S\end{equation} $$

But if there are too many states, the rank of linear equations may be less than number of equations or the susceptibility of pseudo inverse evaluation to changes in state is too much. In short, we want fast-converging approximate non-optimal solution to the problem. Therefore, comes the second way of evaluation.

Assume, Any two probability values of cupcakes. Now keep updating each of these values based on equations, until they converge. In other words, we want steady state solution.

{% highlight python %}

import numpy as np
# S: State Value of cupcake_flavor : Its probability of buying the flavor. 
# P: Transition Matrix : from cupcake_flavorA to cupcake_flavorA, what is probability of transition.
cupcakes = ["choco","sberry"]
S = {"choco": 0, "sberry":1}
P = {"from_choco":{"to_choco": 0.8, "to_sberry" : 0.2 }, 
     "from_sberry": {"to_sberry": 0.1, "to_choco": 0.9}}
for i in range(5):
    choco_future_stateval, sberry_future_stateval = 0,0;
    
    choco_future_stateval = S["choco"]*P["from_choco"]["to_choco"]+\
                            S["sberry"]*P["from_sberry"]["to_choco"];
    
    sberry_future_stateval =  S["choco"]*P["from_choco"]["to_sberry"]+\
                              S["sberry"]*P["from_sberry"]["to_sberry"];
    S["choco"]=choco_future_stateval
    S["sberry"] = sberry_future_stateval
    print ("(Pchoco,Psberry) = (%0.5f,%0.5f)"%(S["choco"],S["sberry"]))

print ("Steady state values of (Pchoco,Psberry) = (%0.5f,%0.5f)"%(S["choco"],S["sberry"]))

{% endhighlight %}
> **Output**  
> (Pchoco , Psberry) = (0.90000 , 0.10000)  
> (Pchoco , Psberry) = (0.81000 , 0.19000)  
> (Pchoco , Psberry) = (0.81900 , 0.18100)  
> (Pchoco , Psberry) = (0.81810 , 0.18190)  
> (Pchoco , Psberry) = (0.81819 , 0.18181)  
> Steady state values of (Pchoco,Psberry) = (0.81819,0.18181)

As we have now established some background of Markov Chain, we will try to wrestle with MDP (Markov Decision Process). Unlike the previous examples MDP's don't estimate steady state probability, they estimate steady state value function. Think of it as a number (not necessarily between 0 and 1) which tells you how rewarding the state is to be. Value Function in steady state tells you average reward you get by traveling anywhere from the given state.

We will now introduce 3 new jargons in blog.

**State Value Function**  $V(s)$  
State Value (in steady state) tells us average reward from a given state. We can compare average rewards of two states based on this value. It can also be looked upon as gist of rewards sprouting out of current state. It possesses markovian property which means, that we can safely cut of calculations of states which are not directly reachable to the current state.  

**Action Value Function**  $Q(a|s)$  
Action Value (in steady state ) tells us average reward if we take a particular action from a given state. This is also markovian in nature. We also call them Q values. We will soon look in the details when we study Q Learning and SARSA update.  

**Discounted Reward or Return**  $G(s)$  
It is weighted sum of rewards, with weights being decaying exponential with decay factor of $\gamma $. If we are traveling in markovian mini-world, then journey can be summarized by traits of state, action and reward.  
Footprints : $s_1,a_1,r_1,s_2,a_2,r_2,s_3,a_2,r_3, \cdots$  

$$G(s_2) = r_2+\gamma r_3 + \gamma^2 r_3 + \cdots $$

> Please, do not associate reward with state, reward is always associated with action in $99\%$ cases.  

Typically we take $\gamma$ as $\geq 0.9$. We can have two extreme cases, based on value of $\gamma$.  

***Case I***   ( $\gamma = 0$ )  
If $\gamma = 0 $ , we become like ***no-brainer*** policy, which we had described earlier.  

***Case II*** ($\gamma = 1$)  
If we make $\gamma = 1$, then we make the reward as sum of all future rewards it will ever have, till it hits one of the terminal state.  


> Note that there is a problem with keeping $\gamma = 1$, In case there are loops in the state transition (i.e, loop means it can come back to the same state again for any state), than it give unstable and inaccurate values. 


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



### Model Free RL Approaches
Model Free implies we are not having preconceived ideas of environment. The only way to know is to act as stated in the previous paragraph. This means, at any state, we only know what all actions I can take, but don't know where I will land up in. 

> **Why do we use only action value function in control problem in pure Model Free Approaches ?**  
>  
> Does the term Model-Free ring a bell? The answer to the question lies in definition of "Model-Free Approaches". If you don't know which state you land up in, you cannot update state value Function. Off course you can estimate State value function (estimate, how rewarding a state is), but you can't use it to decide what action to take. Action values are must for that.  
  
This brings us to two major categories in which Model-Free Approaches are divided. Namely Model-Free Prediction and Model-Free Control. If you take analogy of game of thrones, prediction is like telling how rewarding each house(king's landing, winter-fell) is. This involves state value function estimation. On the other hand, control problem involves finding out what action should be performed from a given state, such as taking a ride on dragon, going through white harbor in ship,etc. This involves knowing action values.  
So, if you want to apply RL in real world, you should first identify it's components as prediction or control problem. If you don't have this clarity, than you will get confused, as to what needs to be applied.  
  
> **What is Episode ? Can we learn for never-ending run ?**  
> Episode is agent's historic information from start to terminal state. Video games are closer to the meaning, wherein you have Super-Mario going from start to the palace. 
> Can we learn without terminal states(non-episodic RL problems)? Answer is yes (that's precisely why we have TD Update), this can be done if the there are immediate rewards and penalty given in the environment, not just on reaching terminal state (which may not be present).


#### Monte Carlo Policy Evaluation
Let us say you are playing a video-game with different states like farm, sea, island, forest, etc. You need to give ranking to these states based on danger. How will you do this?
> Farm $\rightarrow$ Sea $\rightarrow$ Island $\rightarrow$ Forest $\rightarrow$ Dead  
> Sea $\rightarrow$  Island $\rightarrow$ Forest $\rightarrow$ Mountain $\rightarrow$ Forest $\rightarrow$ Dead  
> Forest $\rightarrow$ Sea $\rightarrow$  Island $\rightarrow$ Sea $\rightarrow$ Island $\rightarrow$ Forest $\rightarrow$ Dead

As you can see that death is always preceded by forest. This means forest is definitely dangerous. This intuition can be mathematically thought by MC Updates.

![ Monte Carlo Update ](/assets/img/montecarloupdate.gif)

In the above image, you have visited the farm two times(N=2). and each time you have some discounted reward$(G_1,G_2)$. Then monte carlo estimation for that state will be $(G_1+G_2)/2$. It is that simple.  
  
We keep maintaining the discounted Reward sum ($S(s)$) and total visit count ($N(s)$) for each state and find the average discounted reward($S(s)/N(s)$). That's what Monte Carlo Algorithm is.

![ Monte Carlo Algorithm ](/assets/img/montecarloalgo.png)

We can get rid of N(s) by one simple trick. That is by using ***Moving Average***.  This means, let us update the V(s) by moving $\alpha$ percentile in the direction of difference $(G_t-V(s))$.  

$$V(s) = V(s) + \alpha\overbrace{(G(s)-V(s))}^{\Delta V(s)}$$  

$$V(s) = \alpha G(s) + (1-\alpha )V(s)$$ 

Do you find any limitations of this approach?  
* First of all, The updates are only possible when the episode ends.  
* It needs to keep a huge backup(non-deterministic length), if the episode is very long. This can be a problem in multi-agent scenario.  
* Cannot work for never-ending learning scenarios. The episodic nature of problem is must, unless you have approximations to this algorithm. 


If you are alert, one question which is likely to pop up in your head is that , do we need 100 or 1000 states ahead of me to evaluate $G_t(s)$? can we limit $G_t(s)$ ?  This very question brings us to a better mechanism of policy evaluation called Temporal Difference Updates. 


#### Temporal Difference Policy Evaluation

As we observed earlier, Remy has very small short-term memory. Can we somehow require lesser terms to evaluate $G_t(s)$ ?

$$G_t(s_t) = R_{t+1}+\gamma R_{t+2}+ \gamma^{T-2}\cdots R_{T-1} $$

We can write reward for next state as 

$$G_{t+1}(s_{t+1}) = R_{t+2}+\gamma R_{t+3}+ \gamma^{T-3}\cdots R_{T-1} $$

Therefore we can also write the first equation as 

$$G_t(s_t) = R_{t+1}+\gamma \overbrace {\left\lbrace R_{t+2}+ \gamma^{T-3}\cdots R_{T-1} \right\rbrace}^{G_{t+1}(s_{t+1})} $$

and we all know we want to update value function, closer to discounted reward $(G\leftarrow V)$, therefore our new update for TD Learning becomes 

$$V(s_t)= V(s)+\alpha(\hat{G_t}-V(s))$$

$$V(s_t)= V(s_t)+\alpha(R_{t+1}+\gamma V(s_{t+1})-V(s_t))$$

where we are approximating the  discounted reward $(G_t)$ with $R_{t+1}+\gamma V(s_{t+1})$. This approach updates the value just after the agent moves out into next state. It requires only single state backup and can be useful. This is very obvious alternative to Monte-Carlo updates which require complete episode info for updating the state value function.

![ Temporal Difference Updates ](/assets/img/td-update.png)

In both the approaches we have only looked into, how to evaluate a policy. But the more important task for RL is to learn a policy. This requires us to meddle with Q values of action. 
  
&nbsp;&nbsp;  
&nbsp;&nbsp;  

### Model Free Control
&nbsp;&nbsp;  




| Monte Carlo Updates | Temporal Difference Updates |
|----------------------|------------------|
|a|b|


| Model-Free Prediction | Model-Free Control |
|---------------------- | ------------------ |



## Policy Gradient Methods
Unlike previously learned approaches, Policy gradients don't learn q values. They directly learn policy which more obvious choice. Learning policy mean's learning which action to take. The biggest disadvantage of Deep Q Learning was that it was complicating the problem of learning a policy by adding additional constaint of learning manifold of action values, which doesn't converge easily in many cases. 

> **Read this Karapathy's Blog on PG Methods**  
> I generally don't like to detour readers with blog links. But the simplicity of blog allows me to skip introduction of PG Approaches quite efficiently. This blog is perfect intro to Policy Gradient Approaches.   
>  [http://karpathy.github.io/2016/05/31/rl/](http://karpathy.github.io/2016/05/31/rl/)

If you have read the above blogs, then you should have understood the intuition behind P.G Approach. The difficult with such a model is that, for more complex scenarios, it fails to build an intuition about state representation itself as a mean to explore and learn intelligently. This calls for actor-critic methods where critic is doing state value estimation and actor is taking a call from critic as to how good was its action in current state.  
&nbsp;&nbsp;  
&nbsp;&nbsp;      
### Advantage Actor Critic (A2C)
&nbsp;&nbsp;  
&nbsp;&nbsp;  
&nbsp;&nbsp;  

![Director-Actor Analogy](/assets/img/actorcritic.png)

&nbsp;&nbsp;  
&nbsp;&nbsp;  

Remy, the rat has decided to get away from rat-race and try his luck in acting. Turns out that he has some amazing skills, but the director is quite critical about the role. He keeps instructing Remy as to how to act in this situation(aka state). This director is learning how each situation/state will be (with respect to Remy), and it keeps modifying based on acting skills and performance of Remy in different situation/states. Actor-Critic Methods are just like rat-director pair, where director/critic estimates how good a state is (w.r.t remy's acting) and actor improves the policy(action to take) based on positive/negative feedback from director. If the director gives angry expressions( negative advantage) , he looses the confidence(positive gradient descent) in his policy, and tries to choose different action. But if director is happy, he strengthens his action for current scene.




*Jaley is a storyteller, meme-maker, and so called data scientist, who is too hippy to be serious about anything. (Serious about being not-soo-serious). He believes that he has magical powers to transform nerdy topics into town gossip.*


{% comment %} 
>You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.
>
>To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

	{% youtube oHg5SJYRHA0 200 400 %}
{% endcomment %}

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
