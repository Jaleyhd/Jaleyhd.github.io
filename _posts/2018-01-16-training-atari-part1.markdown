---
layout: post
title:  "Training Atari Game by OpenAI Baselines"
subtitle: "Deep RL Tutorial part-6"
date:   2018-01-16 20:15:00 +0530
comments: true
published: false
permalink : /deeprltutorial/part6/
categories: reinforcement-learning
description: "OpenAI's humongous contribution towards opensource community has been receiving accolates from round the corner. The gym environment and baselines python module has been a wonderful contribution for gaming community, aiming to develop better games. In this tutorial, we will try to understand OpenAI Baseline's implementation of synchronous advantage actor critic"
---

*If you are reading this blog, than we assume that you know Q Learning in theory atleast. If not you can revisit the blog later.*

I assume you have had nightouts for CS(Counter-Strike) and Dota during college days. I never used to like the game, as I felt that it is a waste of time. What if we can ask our bot to play games instead of us. Such hackathons can be like feasty mental food for nerds. They can now play games better than you, because they have trained better bots than you :p



## Rules of the Game
![Atari breakout animation](/assets/img/atari-breakout1.gif)

The game is pretty straightforward. You have only 3 choices at any point of time. To move left, to move right or to stay still. You just need to avoid the ball from hitting ground. This is simple for human to understand, but a machine , needs to understand cause of reward or penalty. 


| Parameters | Value |
|------------|-------|
|Observation | game display of size (210, 160, 3) |
|Action | {0 : no-op, 1 : fire, 2 : left, 3 : right}|
|Info/State| {'ale.lives': 0 or 1} |

### How to use Gym Environment

OpenAI Gym environment provides host of options. To import any gym environment, all you need to do is : 

```python
import gym
env = gym.make('BreakoutNoFrameskip-v4')
```

'-v4' is the latest version of the game without frame skipping.