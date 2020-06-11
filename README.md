# Sensitivity to Calibrated Parameters

This repository contains the replication files for the paper "Sensitivity to Calibration" by Thomas H. Jørgensen. 
The paper can be found [here](https://www.ifs.org.uk/uploads/CWP1620-Sensitivity-to-Calibrated-Parameters.pdf).

The replication package contains two folders. 
+ "GP2002" contains all code for replication of the analysis in Gourinchas and Parker (2002).
+ "Oswald" conatins all code for the sensitivity analysis of the study by Oswald (2019).

## The Sensitivity Measure
I focus on situations in which interest lies in estimating a $K\times1$
vector of parameters, $\theta$, given some $L\times1$ vector of
calibrated parameters, $\hat{\gamma}$. Interest may then be in using
these estimates to subsequently analyze different model outcomes and
predictions. While the proposed sensitivity measure has general applicability,
I will focus attention on estimation of dynamic economic models because
such models typically are time-consuming to estimate.

I assume that the estimation approach employed is of the form
\[
\hat{\theta}=\arg\min_{\theta\in\Theta}g_{n}(\theta|\hat{\gamma})'W_{n}g_{n}(\theta|\hat{\gamma})
\]
where $g_{n}(\theta|\hat{\gamma})=\frac{1}{n}\sum_{i=1}^{n}f(\theta|\hat{\gamma},\mathbf{w}_{i})$
is some $J\times1$ vector valued function of the parameters and data,
$\mathbf{w}_{i}$ for $i=1,\dots,n$, specified by the researcher.
$W_{n}$ is a $J\times J$ positive semi-definite weighting matrix.
When estimating dynamic economic models, evaluating $g_{n}(\theta|\hat{\gamma})$
typically involves solving some model numerically. I assume that the
objective function satisfies standard regularity conditions and abstract
from any numerical approximation error associated with solving the
model.\footnote{see e.g. \citet{NeweyMcFadden1994}.} In particular,
I assume that there exists unique population parameters $\theta_{0}$
and $\gamma_{0}$ such that $g(\theta_{0}|\gamma_{0})\equiv\mathbb{E}\left[f(\theta_{0}|\gamma_{0},\mathbf{w}_{i})\right]=0$.
