---
layout: post
title:  "Topic Modelling"
subtitle: "Uncovering the underlying Structure"
date:   2018-01-08 19:20:38 +0530
comments: true
categories: jekyll update
---
Imagine that, its early morning and you are reading newspaper just dropped by delivery guy. Now while sipping a hot cup of tea, you get an idea. Why can't I organize the articles by cool ML Techinques? The Caffaine pumped man, goes online, collects all Newspaper articles and wants them to be grouped. Thats where LDA comes as a savior. In this case, a newspaper article is document (for LDA) and what we want to do is assign a topic to each of the document. For example 'Novak Djokovic' retiring comes under Sports section and 'Hillary vs Trump' comes under politics section.

Let us consider a scenario where there are n simultaneously occuring events. Each event has m catagories namely $[c_1,c_2,\cdots c_m]$ with probability $[p_1,p_2,\cdots p_m]$ respectively. Now our objective is to find the probability of category $c_1$ occuring $n_1$ times,$c_1$ occuring $n_2$ times, and so on. In other words $[c_1,c_2,\cdots c_m]$ occuring $[n_1,n_2,\cdots n_m]$. Here there two intuitive lemmas/observed constrains are as stated bellow :

$$\sum\limits_{i=0}^{i=m}n_i=n$$

$$\sum\limits_{i=0}^{i=m}p_i=1$$

This joint probability (which is our objective) is given by equation bellow :

$$p(n_1,n_2,\cdots c_m)=\cfrac{\Gamma(n_1+n_2 + \cdots n_m+1)}{\Gamma(n_1+1)\Gamma(n_2+1)\cdots \Gamma(n_m+1)}p_1^{n_1} p_2^{n_2} \cdots p_m^{n_m}$$

where $\Gamma(n)=(n-1)!$ or in other words  $\Gamma(n)=(n-1)(n-2)\cdots 1$.
Wait a second. Does this equation look familiar? I know its too long, but doesn't it remind of some other equation ? Well . . . you guessed it right. It looks very similar to binomial distribution. If there are only two catagories. Here head occurs n1 times and tail occurs n-n1 times The equation becomes

$$p(n_1,n-n1)=\cfrac{\Gamma(n+1)}{\Gamma(n_1+1)\Gamma(n-n_1+1)}p_1^{n_1} p_2^{n-n_1}$$

or simplifying it further it becomes

$$p((n_1)_H)= ^nC_{n_1} p_1^{n_1} p_2^{n-n_1}$$

Here $p((n_1)_H)$ is nothing but probability of getting $n_1$ heads out of n trials.

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
