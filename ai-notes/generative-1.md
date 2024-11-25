# Describing The Problem

In a similar way to the conditional problem, in a generative problem we are given some data and we want to use this data to make a decision. In the generative problem, either we cannot or do not want to split the input data into input variables and target variables, as we want to model the entire distribution. Similar to the conditional problem, we treat the dataset as given, without any influence on its generation.

We can describe the dataset mathematically as the realization of a collection of $N$ random variables $X_i$:

$$
\begin{align}
D &= \{ X_1 = x_1, ... , X_N = x_N\} \\
&= \{ X_i = x_i \}, i=1...N \\
&= \{ x_i \}
\end{align}
$$

To note, $X_i$ can be single or multidimensional, and it can be discrete or continuous. We use the repeated $X$ symbol, because we usually think of each data point as drawn from the same state space (ie $X_t$ is a stochastic process), but the data points could be from different spaces and processes - as long as they are related to the variable needed for making the final decision. We also usually (but not always) treat the $X$s as independent and identically distributed (iid), which significantly simplifies the problem.



In order to make a decision based on the data, we will need to consider the decision making process, ie what are the objectives of the decision. For example, we may want to have the highest expected outcome in $X$, or the highest expected outcome of a function of $X$, or minimize the risk of very low values of $X$. Alternatively, we may just want to generate samples of $X$. However, it is often both possible and useful to separate the modeling into 2 steps:

1. Model the probability distribution of variable $X$ given the data $D$
2. Using the probability distribution, apply a policy to make a decision.

In some cases, for example when we are making multiple decisions in a row and when the system is affected by the decision(s) we make, we cannot make this separation. But in other cases, this separation lead to simplification and flexibility.

Hence, we will look at the problem of calculating:

$$
\begin{align}
P(&X = x \mid \{X_1 = x_1, X_2 = x_2, ... X_N = x_N \}) \\
&= P(x \mid \{ x_i \}) \\
&= P(x \mid D)
\end{align}
$$

## Applying a Model

As in the conditional case, the first step is conditioning on a model M:

$$
P\left(x \mid D\right) = \sum_M P\left(x \mid M\right) P\left(M \mid D\right)
$$

and then applying Bayes rule:

$$
P(x \mid D) = \frac{\sum_M P(x \mid M) P(D \mid M) P(M)}{\sum_M P(D \mid M) P(M)}
$$

For a parameterized model (ie a model where the number of parameters does not grow with the number of data points), the model can be replaced by the parameters. In the discrete case:

$$
P(x \mid D) = \frac{\sum_i P(x \mid w_i) P(D \mid w_i) P(w_i)}{\sum_i P(D \mid w_i) P(w_i)}
$$

and continuous:

$$
P(x \mid D) = \frac{\int P(x \mid w) P(\{ x_i \} \mid w) P(w) dw}{\int P(\{x_i\} \mid w) P(w) dw}
$$

Similar to the conditional case, we have 3 options:

1. Choose a parameterization that has conjugate priors.
2. Attack the integrals numerically.
3. Assume that some of the terms under the integral are sharply peaked in $w$-space and that the maximal values can be taken instead of the whole integral.

## Maximizing the Likelihood

### Maximum a Posteriori Likelihood

Assuming $P(\{x_i\} \mid )$ P(w)$ is sharply peaked in w-space. By substituting

$$
P( \{ x_i \} \mid w) P(w) = \frac{1}{f( \{ x_i \})} \delta(w - w_{MAP})
$$

we get

$$
P(x \mid w) = P(x \mid w_{MAP})
$$

Hence, the problem becomes:

1. Choose a suitable parameterization of the distribution $P(\{ x_i \} \mid w)$ and the prior $P(w)$.
2. Find the maximum of $P( \{ x_i \} \mid w) P(w) $ with respect to $w$.
3. Plug $w_{MAP}$ back into the $P( x \mid w).

### Maximum Likelihood

Assuming $P(\{ x_i \} \mid w)$ is shareply peaked in $w$-space. By substituting

$$
P( \{ x_i \} \mid w) = \frac{1}{f( \{ x_i \})} \delta(w - w_{MLE})
$$

we get

$$
P(x \mid w) = P(x \mid w_{MLE})
$$

Hence, the problem becomes:

1. Choose a suitable parameterization of the distribution $P(\{ x_i \} \mid w)$.
2. Find the maximum of $P( \{ x_i \} \mid w) $ with respect to $w$.
3. Plug $w_{MLE}$ back into the $P( x \mid w).

Note: the modeling step here is very similar to that in the 'Maximum Generative Likelihood' section for the conditional distribution.
