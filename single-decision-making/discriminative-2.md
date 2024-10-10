# From a Likelihood to Error Functions

## Continuous Target Variable(s)

We are looking to find $w_{MDLE}$ that maximizes the function $P( \{ t_i \} | \{x_i \}, w )$.

If we assume that $t$ is continuous and given by a deterministic function of $x$ with additivie Gaussian noise. ie

$$
t = y(x, w) + \epsilon
$$

then

$$
P(t \mid x, w, \sigma^2) = N(t \mid y(x, w), \sigma^2)
$$

By assuming that the data points are iid, the likelihood becomes

$$
P(\{t_i\} | \{x_i\}, w, \sigma^2) = \prod_{n=1}^{N} N(t_n | y(x_n, w), \sigma^2)
$$

where

$$
N(t_n | y(x_n, w), \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(t_n - y(x_n, w))^2}{2 \sigma^2} \right)
$$

Equivalent to maximizing the likelihood is minimizing the negative log likelihood. This becomes

$$
\ln P({t_i} | {x_i}, w, \sigma^2) = \frac{N}{2} \ln(2\pi) + \frac{N}{2} \ln \sigma^2 + \frac{1}{2 \sigma^2} \sum_{n=1}^N \left( t_n - y(x_n, w)\right)^2
$$

This has helpfully separated the maximization of the likelihood into two steps: 1) maximizes with respect to $w$, 2) maximize with respect to $\sigma$.

For the $w$ maximization, the first two terms are constant and drop out of the maximization. Hence, we can find the values of $w$ that maximize the likelihood by minimizing the least squares error function:

$$
E(w) = \sum_{n=1}^N \left( t_n - y(x_n, w)\right)^2
$$

This conclusion does not depend on the form of $y(x, w)$, just that it is a deterministic function of $x$, and that the noise term is Gaussian.

If we only need the expected value of $t$, we do not have to go on and find the value of $\sigma$.

## Discrete Target Variable(s)

If we assume our data is discrete, and that each input variable $x_n$ corresponds to a single category $C_n$, then we can use 1-of-K coding:

$$
P(t_n \mid x_n, w) = \prod_{k=1}^{K} P(C_k \mid x_n, w )^{t_{nk}}
$$

Then, with iid data, the likelihood is:

$$
P(\{ t_i \} \mid \{ x_i \}, w) = \prod_{n=1}^N \prod_{k=1}^{K} P(C_k \mid x_n, w)^{t_{nk}}
$$

The error term is the negative log likelihood:

$$
E(w) = - \sum_{n=1}^N \sum_{k=1}^K t_{nk} \ln P(C_k \mid x_n, w)
$$

This is the cross-entropy error function. By minimizing this function, we are maximizing the likelihood. Once again, we have made limited assumptions in deriving this error function.

In this case, we have to pick a parameterization for $P(C_k \mid x_n, w)$ that satisfies $\sum_{k=1}^{K} P(C_k \mid x, w) = 1$. One approach to this is to use the soft-max function, thus allowing the parameterization to go over all the real numbers:

$$
P(C_k \mid x, w) = \frac{\exp( y_k(x, w))}{\sum_j \exp( y_j(x, w))}
$$

In this case, the error term becomes

$$
E(w) = - \sum_{n=1}^N \left( \sum_{k=1}^K t_{nk} y_{nk} - \ln \sum_j \exp(y_{nj}) \right)
$$

having shortened $y_k(x_n, w) = y_{nk}$, referred to as the 'logits'.
