---
layout: post
title:  "Getting rid of environment"
subtitle: "Deep RL Tutorial part-3"
date:   2018-01-12 20:15:00 +0530
comments: true
published: true
permalink : /deeprltutorial/part3/
categories: reinforcement-learning
description: "We are getting very close to training Atari games. Whatever we learning in part-2 involves extreme assumptions about environment. Here we will try and learn techniques which can work in any environment"
---

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
