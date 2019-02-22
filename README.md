# README

The following text by [Jens Brage](jens.brage@noda.se) was written using [Upmath: Markdown & LaTeX Online Editor](https://upmath.me).

## The alternating direction method of multipliers

The NODA STORM tracker, here reduced to its essentials as part of the project Netflexible Heat Pumps, serves to make the aggregated supply and demand of a large number of distributed energy resources track a desired plan. To this end, it is necessary to perform model predictive control over a horizon of days at a resolution of a fraction of an hour. Distributed convex optimization in the guise of the *alternating direction method of multipliers* (ADMM) offers a way to address the large-scale nature of the problem; see, for reference, [B2010].

The implementation of the NODA STORM tracker follows [K2012] and [K2013], which addresses the problem of *optimal power scheduling* in a network of devices interconnected by local nets, and provides a solution along the lines of the algorithm of *prox-average message passing*. The approach generalizes the problem of *optimal exchange* in a star-shaped network of devices interconnected by a global net, the difference being the amount of bookkeeping necessary to describe the problem and its solution.

The exchange problem is to minimize

$$\sum_{i=1}^{N} f_{i}(x_i)$$

subject to

$$\sum_{i=1}^{N} x_{i} = 0$$

with variables $$x_{i} \in \bold{R}, i = 1, ..., N$$, where $$f_{i} : \bold{R} \to \bold{R} \cup \{+\infty\}$$ represents the cost function for device $$i$$, or what amounts to the same thing, to minimize

$$\sum_{i=1}^{N} f_{i}(x_i) + g(\sum_{i=1}^{N}(x_{i}))$$

where $$g : \bold{R} \to \bold{R} \cup \{+\infty\}$$ encodes the constraints, that is, $$g(0) = 0$$ and otherwise $$g(x) = +\infty$$.

Written in ADMM form, the exchange problem becomes to minimize

$$\sum_{i=1}^{N} f_{i}(x_i) + g(\sum_{i=1}^{N}(z_{i}))$$

subject to

$$x_{i} - z_{i} = 0, i = 1, ..., N$$

with variables $$x_{i}, z_{i} \in \bold{R}$$. The corresponding ADMM algorithm is

$$x_{i}^{k+1} := \text{argmin}_{x_{i}}(f_{i}(x_{i}) + (y^{k})^{T} x_{i} + (\rho / 2) ||x_{i} - z_{i}^{k}||_{2}^{2})$$

$$z^{k+1} := \text{argmin}_{z}(g(\sum_{i=1}^{N}) - (y^{k})^{T} x_{i} + (\rho / 2) ||x_{i}^{k+1} - z_{i}||_{2}^{2})$$

$$y_{i}^{k+1} := y_{i}^{k} + \rho (x_{i}^{k+1} - z_{i}^{k+1})$$

with variables $$y_{i} \in \bold{R}^{n}, i = 1, ..., N$$, where $$\rho > 0$$ is a penalty parameter, and the $$y_{i}$$ are Lagrange multipliers. Solving for $$z$$, the algorithm can be simplified to

$$x_{i}^{k+1} := \text{argmin}_{x_{i}}(f_{i}(x_{i}) + (y^{k})^{T} x_{i} + (\rho / 2) ||x_{i} - (x_{i}^{k} - \bar{x}^{k})||_{2}^{2})$$

$$y^{k+1} := y^{k} + \rho \bar{x}^{k+1}$$

with variable $$y \in \bold{R}^{n}$$, where $$\bar{x}$$ is the average of $$x_{i}, i = 1, ..., N$$.

The variable $$y^{k}$$ converges to an optimal dual variable that can be interpreted as a set of optimal clearing prices for the exchange of commodities between device $$i = 1, ..., N$$, with $$(x_{i})_{j}$$ the amount of commodity $$j$$ *recieved* by device $$i$$ from the exchange; see, for example, the classical works by [W1896], [A1954] and [U1960(1), U1960(2)].

The economic interpretation carries over to the power sheduling problem, with the prox-average message passing algorithm realising a multi-market negotiation process.

## References

[A1954] K. J. Arrow and G. Debreu, "Existence of an equilibrium for a competitive economy," *Econometrica: Journal of the Econometric Society,* vol. 22, no. 3, pp. 265-290, 1954.

[B2010] S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein, "Distributed optimization and statistical learning via the alternating direction method of multipliers," *Foundations and Trends in Optimization,* vol. 3, no. 1, pp. 1-122, 2010.

[K2012] M. Kraning, E. Chu, J. Lavaei and S. Boyd, "Message passing for dynamic network energy management," arXiv:1204.1106v1 [math.OC], 2012.

[K2013] M. Kraning, E. Chu, J. Lavaei and S. Boyd, "Dynamic network energy management via proximal message passing," *Foundations and Trends in Optimization,* vol. 1, no. 2, pp. 70-122, 2013.

[U1960(1)] H. Uzawa, "Market mechanisms and mathematical programming," *Econometrica: Journal of the Econometric Society,* vol. 28, no. 4, pp. 872-881, 1960.

[U1960(2)] H. Uzawa, "Walras' tâtonnement in the theory of exchange," *The Review of Econometric Studies,* vol. 27, no. 3, pp. 182-194, 1960.

[W1896] L. Walras, *Éléments d'économie politique pure, ou, Théorie de la richesse sociale.* F. Rouge, 1896.