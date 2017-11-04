# Dirichlet process mixture models
This repository implements a Dirichlet process mixture model (dpmm). It assumes the data to be a mixture of multinoulli vectors. In [this project](https://robromijnders.github.io/dpm/) I implemented the dpmm assuming a mixture of Gaussians on the data. That project explains much of the workings of the Dirichlet process. So on this readme, we will only discuss the particulars of the multinoulli model and some insights on Gibbs sampling.

# What is a Multinoulli and why should we care?
We can think of a multinoulli distribution as a series of Bernoulli distributions. For example, we do quality control of car parts. Then we have questions like "Is the rear light broken yes/no?", "Does the engine make a sound yes/no", "Are the tires at incorrect pressure yes/no?", "is the oil at incorrect level yes/no?" and "does the hood have a scratch yes/no?". Such measurements is just a series of 0's and 1's. Like <img alt="$x = (0, 1, 0, 0, 1)$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/5e2059f1f01f732f7d08b2460a235390.svg?invert_in_darkmode" align=middle width="114.417765pt" height="24.6576pt"/>. Each question has a probability of answered yes/no. In other words, each variable is Bernoulli distributed. Then we say <img alt="$x$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width="9.3951pt" height="14.15535pt"/> is Multinoulli distributed.

An example of a probability vector for a Multinoulli is <img alt="$\pi = (0.3, 0.9, 0.3, 0.8, 0.1)$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/5ece9892e668eaf7d9609c29934a86dc.svg?invert_in_darkmode" align=middle width="178.910655pt" height="24.6576pt"/>. You may read this is as

  * With probability 0.3, the rear light is broken
  * With probability 0.9, the engine makes a strange sound
  * With probability 0.3, the tires are at incorrect temperature
  * With probability 0.8, the oil level is incorrect
  * With probability 0.1, the hood has a scratch

(Note that the probabilities do not sum to 1! Each question associates with a probability of being yes or no. This contrasts with __Multinomial__ distributions, where indeed the probabilities must sum to 1. That would be interpreted as _only one question will be yes_, which is in contrast with our model.)

## How to imagine a mixture model of Multinoulli's?
In a mixture model, we imagine the cars in our garage to have problems in some number of clusters. Each cluster associates with a Multinoulli distribution. In our example, we may observe that cars with broken rear lights, usually also have a scratch. This corresponds to the cluster <img alt="$\pi = (0.9, 0.1, 0.2, 0.1, 0.8)$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/9e31b85126a8cd87d254767bdf2afaf1.svg?invert_in_darkmode" align=middle width="178.910655pt" height="24.6576pt"/>. Or the cars with engine trouble also usually have bad oil levels, corresponding to a cluster <img alt="$\pi = (0.3, 0.9, 0.3, 0.8, 0.1)$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/5ece9892e668eaf7d9609c29934a86dc.svg?invert_in_darkmode" align=middle width="178.910655pt" height="24.6576pt"/>

# Collapsed Gibbs sampling
So how do we go about working with this model?

The important variable in our model are the cluster assignments. The variable, <img alt="$z_i$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width="12.295635pt" height="14.15535pt"/> is an integer that informs us what cluster the point <img alt="$x_i$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode" align=middle width="14.045955pt" height="14.15535pt"/> belongs to. For clarity, this picture summarizes a graphical representation of our model (from Murphy 25.2, page 887).
![graphical model](www.link.com)

As we only care about the cluster assignments, we can integrate out the other parameters. That way, the Gibbs sampler only needs to sample the <img alt="$z_i$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width="12.295635pt" height="14.15535pt"/>. More formally, we can get away by only sampling
<p align="center"><img alt="$$p(z_i = k|z_{-i}, x, \alpha, \lambda)$$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/82914e84021a2e9ba2fb3ca987005066.svg?invert_in_darkmode" align=middle width="144.60237pt" height="16.438356pt"/></p>

# Gibbs sampling results in samples from a posterior. Now what?
Gibbs sampling results in samples from the posterior distribution over <img alt="$z$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width="8.367645pt" height="14.15535pt"/>. These samples allow us to calculate properties of this posterior distribution. A main result in the literature on Monte Carlo sampling is that as we have have samples from a distribution, we can use it to estimate expected values of functions of this distribution. Like so

<p align="center"><img alt="$$ E[f(X)] \approx \frac{1}{N}\sum_n f(x_n) $$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/7112ffad8cb0e714c6b18fd64971859d.svg?invert_in_darkmode" align=middle width="170.7585pt" height="40.618545pt"/></p>

For Markov Chain Monte Carlo, we need to take note that <img alt="$x_n$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/d7084ce258ffe96f77e4f3647b250bbf.svg?invert_in_darkmode" align=middle width="17.52102pt" height="14.15535pt"/> are unbiased and i.i.d. samples from <img alt="$p(X)$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/f99913a4cacad8463355fd388c8fbc64.svg?invert_in_darkmode" align=middle width="35.96472pt" height="24.6576pt"/>. That has two consequences

  * __Burn in__: During the burn in period, we omit the samples from the Markov chain. The initial samples depend too much on the choice of the initial conditions of the Markov chain. Therefore, they are not _unbiased_ samples. After this burn in period, the samples from the Markov chain are valid samples from <img alt="$p(X)$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/f99913a4cacad8463355fd388c8fbc64.svg?invert_in_darkmode" align=middle width="35.96472pt" height="24.6576pt"/>
  * __Sample interval__: Only samples at some interval will be used. For example, we save every k-th sample and omit the intermittent samples. In our Markov chain, subsequent samples are _correlated_. So a sample and its successor might be dependent. Remember that we can only approximate <img alt="$E[f(X)]$" src="https://rawgit.com/RobRomijnders/dpbmm/master/svgs/e3c591e3daeb1d8ac149c94a25ac5e59.svg?invert_in_darkmode" align=middle width="59.726205pt" height="24.6576pt"/> when we have i.i.d. samples. Therefore, we use only the samples at a fixed interval.

## Estimate the number of clusters
We will illustrate this sample procedure by estimating the number of active clusters. In line 61 to 75 of `main_dpm.py`, we track the Monte Carlo samples of measuring the number of clusters. According to the idea of Monte Carlo sampling, the expected number of clusters amounts to just the sample mean of this list.

# How the code works
The code generates random data to illustrate the working of our model. This is the `generate_dataset()` function. You'll notice that the model finds (usually) more clusters than we actually generated. But also mention the number of points assigned to the clusters. Usually, we recognise the generated data in the larger clusters.

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com