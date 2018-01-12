---
layout: post
title:  "Deconstructing Reinforcement Learning"
subtitle: "Its all about policy"
date:   2018-01-09 20:15:00 +0530
comments: true
categories: reinforcement-learning
---

# Why is RL a Game Changer?
When you start binge watching David Silver RL Lectures on a bright Saturday Morning, its hard to not think that there is a whole world out there yet to be explored. Reinforcement Learning is not a new concept. In 1950's Richard Bellman was working hard to explain behavior of time-dynamic systems. He became father of what can be considered a remarkable problem-solving techinique, which is quite popular amongst CS-Algo students. Its **Dynamic Programming**. It is most clever way of reusing computations of optimal substructures, for example in Knapsack Problem. 

In Reinforcement Learning, the substructures are subsolutions. 
It is  The idea of optimality in Markovian processes, specifically Markov Decision processes can be said to foundation to RL Techniques. This contribution by Bellman, in 1957, also known as Bellman Optimality Equation, can explain how cooking pans arrive at steady state temperature after 10 minutes. 

> If you have mercury thermometers at home for measuring fever, you can observe, how mercury comes to steady state. This can be looked as markov process with single state. However, things can be pretty messed up, in case of pan, where different parts of pan(continuous state) are at different temperature, yet at an equilibrium. To explain such behavior in thermodyamics, we can use Bellman Equations.

![Ratatouille Banner](/assets/img/ratatouillebanner.png)

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


# No Brainer Policy
![Immediate Reward](/assets/img/immediatereward.png)

What is the simplest policy can follow in any environment? Well, we can just look at immediate reward and decide our next state. As you can see in the above diagram, such policy would take the yellow path, as it goes by immediate reward. So I guess you know what's the problem. The problem is, that this no-brainer policy doesn't know if there is another path which could eventually make it hit the jackpot. Thats the exact reason we need a dynamic model which takes into account future rewards. We will come back to this problem after a short detour of Markov Chain which is necessary to understand dynamic systems.

# Markov Chain
![Cupcake Markov Chain](/assets/img/cupcakemarkovchain.png)

In 2 broke girls, do you know why the cupcake business kick-started by X and Y did not take off? Well, its because they didn't meet pal of Z from Russia. Ya, you guessed it, Its Markov. Had they met Markov, he would have explained how customers are eating cupcakes. 

Now strawberry cupcake tastes great, but you are unlikely to get addicted to it, on the other hand, chocolate cupcakes, just hit the right spot in brain, releasing all endorphins. Therefore, you will keep buying chocolate cupcakes again and again.

Our target here is to find probability of buying strawberry cupcake opposed to buying chocolate cupcake. And so, we setout a simple experiment. With help of Markov, we track a customer's cupcake pattern and find out the transition probabilities based on counts. Let me explain. 


> ### Constructing Markov Transition model from scratch
> If a customer bought 101 cupcakes in 101 days. He transitioned from chocolate to strawberry only 5 times, kept holding on to chocolate for 45 times. Similarly he transitioned from strawberry to chocolate 25 times, and strawberry to strawberry another 25 times.
>
 $$ P(S | C) = P(C\rightarrow S) = \cfrac{\#C\rightarrow S}{\#C} = \cfrac{\#C\rightarrow S}{\#C\rightarrow S + \#C\rightarrow C} = \cfrac{5}{5+45} =0.1 $$
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

# Learning Policy

![Rat in Map](/assets/img/ratinmap.png)

$$ \pi (s) : S \rightarrow A $$

Our dear rat, is on a mission to discover most rewarding path to cheese. He can either climb up through the curtains and jump on kitchen floor, or he can climb via plants in balcony and enter kitchen from window. To learn the most optimal policy is rat's goal, and is also the goal of Reinforcement Learning. A policy charts out the action to be taken in a given state. Think of it like SIRI in navigation. At each turn(state), SIRI tells you the next best action.

In the next section,  we will learn bellman equations for  Value Iteration as a way of getting best policy.

# Bellman Equations  
*Bellman Optimality Equation for Value Iteration*


If you can see, the above thug life image of our rat, he is currently in state S1, and we need to find a way to update its state values.

$$ V^{(i+1)}(s_1) \leftarrow  \underbrace{max}_{s \in \{s_2,s_3\}} R_{a(s_1,s)} + \gamma  V^{(i)}(s) $$

$$ \pi^{(i+1)}(s_1) = \text{action}\in \{ a_1,a_2 \} \text{ with max future value } \{ V(s_1),V(s_2) \} $$

It means we change our policy greedily based on higher Value. Let us say $R_2+\gamma V(s_2)$ is 40, and $R_3+\gamma V(s_3)$ is 50, then we select action $a_3$ as it leads us to higher value. In short you update Values of states first and then update policy. You keep repeating these steps simultaneously(unlike policy iteration), until the value and policy converges. 

I know, math has started picking up pace, but there is one additional concept which needs to be brought. The outcome of action may be probabilistic. This means, our dear rat may jump from curtain to kitchen, but it can have a crash too. So, these cruel mathematicians added another layer of complexity, with outcome of action being probabilistic. This means same acton a, can lead you to different states probabilistically. For example there a long pit in front of you, the action is jumping, but there is 50-50 chances of either going across the pit, or landing in the pit with mud water. 



> ***Algorithm-1 (Value Iteration with deterministic action output)***
> Initialize V(s) $\forall s \in S$
> i = 0;
> **while** values converge:  
> &nbsp;&nbsp;&nbsp;&nbsp; V(s) \leftarrow R
> 
***Jaley is a storyteller, meme-maker, and so called data scientist, who is too hippy to be serious about anything. He believes that he has magical powers to transform nerdy topics into town gossip.***


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
