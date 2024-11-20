# From an Integral to a Maximization

## Describing the Problem

We are given some data and we want to use this data to make a decision. We focus here on the case where each data point splits into two parts - the input variable(s) and the target variable(s) - and the new decision comes with a new piece of information, the new input variable(s). This is often described as a conditional or discriminative problem.

For now, we treat the dataset as given, where we do not have any influence on its generation. In other problems, we may have input into how the data is generated - we get to choose or influence how to explore the state of possibilities - which turns the problem into making a sequence of decisions. Here, we focus on a static dataset used to make a single decision.

For example,

1. Given a set of audio data labeled with the words in the audio and a new snippet of audio, state what words have been said. The input variables are the audio data and the target variables the spoken words.

2. Given a set of text data and a sequence of words that form a question, state the string of words that forms the answer. The input variables are sequences of words, and the target variables are the words that follow the input sequence.

3. Given the historical price movements of 1000 stocks, make a decision on whether to buy, sell or hold a specific stock. The input variables are the price movements of the stocks up to a specific time, the target variables the price movements after that time (where the specific time can be varied across the historical period covered).

We can describe the dataset mathematically as the realization of a collection of $N$ pairs of random variables, $X_i$ and $T_i$. Three versions of notation, in order of increasing conciseness, are:

$$
\begin{align}
D &= \{ (X_1 = x_1, T_1 = t_1), ... (X_N = x_N, T_N = t_N) \} \\
&= \{ (X_i, T_i) = (x_i, t_i) \}, i=1...N \\
&= \{(x_i, t_i) \}
\end{align}
$$

To note, both $X_i$ and $T_i$ can take single or multidimensional values, and can be discrete or continuous. We use the repeated $X$ and $T$ symbols, because we usually think of each data point as drawn from the same state space (ie $X_t$ and $T_t$ are stochastic processes), but each data point could be from a different space and process - as long as they are related to the variable needed for making the final decision. We also usually (but not always) go even further, and treat the $T$s given $X$s as conditionally independent and identically distributed (iid), which significantly simplifies the problem.

A significant part of the overall problem will be how to represent the data, both conceptually (ie the set of possibilities) and specifically on a computer (ie the number representation of the set of possibilities). For example, written language could be represented as a sequence of words, where each word is mapped to a number, the position of the word in the index. Or each word could be broken up into parts; or each character could be taken separately, or even the binary representation of unicode text could be used. Each of these choices has advantages and disadvantages - but in the end, the practical results are most important, which are often only apparent once a full end-to-end model has been produced. This becomes the art of feature engineering: given a compute and effort budget, what representation is most likely to produce the best decision making model?

In order to make a decision based on the data, we will need to consider the decision making process, ie what are the objectives of the decision. For example, we may be ok working with the expected value of $T$, or want to know the risk of very low values of $T$. However, it is often both possible and useful to separate this into 2 steps:

1. Model the probability distribution of target variable $T$ given the new input variable $X$ and the data $D$
2. Using the probability distribution, apply a policy to make a decision

In some cases, for example when we are making multiple decisions in a row and when the system is affected by the decision(s) we make, we cannot make this separation. But in other cases, this separation leads to simplification and flexibility.

Hence we will look at the problem of calculating:

$$
\begin{align}
P&(T = t \mid X = x, \{(X_1 = x_1, T_1 = t_1), ... (X_N = x_N, T_N = t_N) \})
\\ &= P(t \mid x, \{ x_i, t_i \})
\\ &= P(t \mid x, D)
\end{align}
$$

for values of $t$ within the specified range of the problem.

## Approaches to the Probability Distribution

As a first step, we combine the sum and product rules of probability to condition the probability on a model $M$, and sum over all possible models.

$$
P\left(t \mid x, D\right) = \sum_M P\left(t \mid x, M\right) P\left(M \mid D\right)
$$

We have assumed that all data dependence is taken by the model $M$, leaving the value of $T$ conditionally dependent only on $X$ and the model $M$. In the case of a continuous set of models, the sum becomes an integral. If the model $M$ is given by a fixed set of parameters that does not grow with the number of data points in $D$ then we have parameterized the problem. In some cases, this may not be possible and the introduction of a model is not wanted, or a model with a number of parameters that increases with the number of data points is wanted. The latter case is described by non-parametric models.

By doing this, we have separated the probability into two parts:

1. The probability of the model $M$ given the data $D$
2. The probability of the new result $T = t$ given the model $M$ and $X = x$.

By using Bayes rule to express $P(M \mid D)$ in terms of $P(D \mid M)$ gives:

$$
P(t \mid x, D) = \frac{\sum_M P(t \mid x, M) P(D \mid M) P(M)}{\sum_M P(D \mid M) P(M)}
$$

where $P(M)$ is our prior on the model $M$, and the denominator is often expressed as $P(D) = \sum_M P(D \mid M) P(M)$.

Rather than an abstract set of models $M$, we will usually set our prior to zero for most models, and only non-zero for a specific class of model, parameterized by $w$. For example, we may only want to consider models that have Gaussian noise around a mean, and ignore any other distributions. $w$ can be single dimensional (ie a scaler) or multidimensional (eg a vector). Assuming that $w$ is continuous (although $X$ and $T$ can still be discrete), the above can be rewritten as:

$$
P(t | x, D) = \frac{\int P(t \mid x, w) P(\{ x_i, t_i \} \mid w) P(w) dw}{\int P(\{x_i, t_i\} \mid w) P(w) dw}
$$

The task then is to decide on the parameterization for $P(t | x, w)$, a prior $P(w)$ for the weights $w$, and to evaluate the integrals. 3 possible approaches to this are:

1. Choose a parameterization for $P(t | x, w)$ from a family that has analytic conjugate priors (eg the exponential family), and then choose the conjugate prior distribution for $P(w)$. The integrals can then be evaluated analytically. This gives a closed-form solution for the distribution, which is very cheap to compute (relative to a numerical solution), but the assumption on the distribution $P(t | x, w)$ may be too simplistic.

2. Choose more general forms for the distribution and the prior, and attack the integrals numerically. This can be applied to much wider ranges of distributions, but the numerical integral may be computationally expensive, or even not possible in any realistic time-frame.

3. Assume that some of the terms under the integrals are sharply peaked in $w$-space, and the maximal values can be taken instead of evaluating the whole integral. This can be applied to a wide range of problems, but may not be realistic in cases with small amounts of data.

## An Example: iid Gaussian probabilities

We take an example, where $X = 0$ for all $i$, we assume that $P({t_i} \mid w)$ is one-dimensional and distributed according to a Gaussian distribution, and all $T_i$ are independent and identically distributed. Hence,

$$
P( \{x_i, t_i \} \mid w) = P(\{t_i\} \mid \mu, \sigma^2) = \prod_{i=0}^N  \mathcal{N}(t_i \mid \mu, \sigma^2)
$$

The parameter matrix $w$ is 2 dimensional, $(\mu, \sigma^2)$.

By completing the square, this can be rewritten as:

$$
P(\{t_i\} \mid \mu, \sigma^2) = \frac{1}{(2\pi)^N} \frac{1}{(\sigma^2)^{\frac{N}{2}}} \exp \left( - \frac{N}{2\sigma^2} ( \mu - \bar{\mu})^2 \right) \exp \left( -\frac{1}{2\sigma^2} \sum_i (t_i - \bar\mu)^2 \right)
$$

where $\bar\mu = \frac{1}{N} \sum_i t_i$.

This is in the form of a Normal-Inverse-Gamma distribution over the :

$$
NIG(\mu, \sigma^2 \mid \mu_0, \nu, \alpha, \beta) = 
\frac{\sqrt{\nu}}{\sqrt{2\pi}}\frac{\beta^\alpha}{\Gamma(\alpha)}
\left( \frac{1}{\sigma^2} \right)^{\alpha + \frac{3}{2}}
\exp \left( - \frac{N}{2\sigma^2} ( \mu - \mu_0)^2 \right)
\exp \left( - \frac{\beta}{\sigma^2} \right)
$$

where $\mu_0 = \bar\mu = \frac{1}{N} \sum_i t_i$, $\nu = N$, $\alpha = \frac{1}{2} (N - 3)$ and $\beta = \frac{1}{2} \sum_i (t_i - \bar\mu)^2$.

The first observation is that this will tend towards a $\delta$-function of $\mu$ and $\sigma^2$ as $N \rightarrow \infty$.

We can also look at the general $N$ case when $P(w)$ is distributed according to the conjugate prior distribution of the Gaussian, which (not coincidentally) is also the Normal-Inverse-Gamma.

$$
\begin{align}
\mu_o &\rightarrow \frac {\nu \mu _{0}+n{\bar {x}}}{\nu + N} \\

\nu &\rightarrow \nu + N \\

\alpha &\rightarrow \alpha +{\frac {N}{2}} \\

\beta &\rightarrow 

\end{align}
$$

The posterior predictive distribution, $P(t | x, D)$ becomes a t-distribution:

$$
P(t | x, D) = t_{2\alpha '}\left({\tilde {x}}\mid \mu ',{\frac {\beta '(\nu '+1)}{\alpha '\nu '}}\right)
$$

In general, the t-distribution is much more fat-tailed than a normal distribution - ie the full Bayesian probability for $P(t | x, D)$ is much more fat-tailed than the maximum likelihood estimation. As $N \rightarrow \infty$, $\alpha \rightarrow \infty$ and we regain the normal distribution.

What if the likelihood is not sharply peaked in $w$ space? As long as our prior over $w$ does not vary significantly over the space in which $w$ is not varying, then can still use maximum likelihood.

## Maximizing the Likelihood

In this third case, we also have 3 options:

### Maximum a posteriori

Assuming $P( \{x_i, t_i \} \mid w) P(w)$ is sharply peaked in w-space. By substituting 

$$
P( \{x_i, t_i \} \mid w) P(w) = \frac{1}{f(\{x_i, t_i \})} \delta (w - w_{MAP})
$$

we get

$$
P(t \mid x, w) = P(t | x, w_{MAP})
$$

the maximum a posteriori values (the normalization factors cancel across the two integrals). Hence, the problem becomes:

1. Choosing a suitable parameterization of the distribution $P( \{(x_i, t_i)\} | w)$ and the prior $P(w)
2. Finding the maximum of $P( \{(x_i, t_i)\} | w) P(w)$ with respect to $w$
3. Plugging $w_{MAP}$ back into the parameterization of $P(t | x, w)$. Since we have chosen to model $P( \{(x, t)\} | w)$, we use

$$
P(t \mid x, w_{MAP}) = \frac{P(t, x \mid w_{MAP})}{\int P(t, x \mid w_{MAP}) dt}
$$

to find the final result.

### Maximum Generative Likelihood

Assuming $P( \{ x_i, t_i \} \mid w)$ is sharply peaked in w-space. By substituting

$$
P( \{x_i, t_i \} \mid w) = \frac{1}{f(\{x_i, t_i \})} \delta (w - w_{MLE})
$$

we get

$$
P(t \mid x, w) = P(t | x, w_{MGLE})
$$

the maximum (generative) likelihood values (the normalization factors and the priors cancel across the two integrals). 

Hence, the problem becomes:

1. Choosing a suitable parameterization of the distribution $P( \{(x_i, t_i)\} | w)$
2. Finding the maximum likelihood of this distribution with respect to $w$
3. Plugging $w_{MLE}$ back into the distribution. Since we have chosen to model $P( \{(x, t)\} | w)$, we use

$$
P(t \mid x, w_{MLE}) = \frac{P(t, x \mid w_{MLE})}{\int P(t, x \mid w_{MLE}) dt}
$$

to find the final result.

### Maximum Discriminative Likelihood

In the third approach, we use the product rule to express

$$
P( \{ x_i, t_i \}) = P( \{ t_i \} \mid \{ x_i \}, w) P(\{ x_i \})
$$

and then assume that $P( \{ t_i \} \mid \{ x_i \}, w)$ is sharply peaked in w-space,

$$
P( \{t_i\} \mid \{ x_i \}, w) = \frac{1}{f(\{x_i, t_i \})} \delta (w - w_{MDLE})
$$

then we get

$$
P(t \mid x, w) = P(t | x, w_{MDLE})
$$

the maximum (discriminative) likelihood values (again, the normalization factors and priors cancel across the integrals). Hence, the problem becomes:

1. Choosing a suitable parameterization of the conditional distribution (the 'likelihood') $P( \{ t_i \} | \{x_i \}, w )$
2. Finding the maximum of the likelihood with respect to $w$
3. Plugging $w_{disc}$ back into $P( t \mid x, w_{disc})$.

While the maximum generative likelihood and maximum discriminative likelihood approaches are related, the discriminative assumption is a stronger assumption on the conditional distribution.
