---
title: "Convergence of Heavy-Ball Method and Nesterov's Accelerated Gradient on Quadratic Optimization"
collection: notes
permalink: /notes/2018/Momentum
date: 2018-09-07
excerpt: We will study the proof of convergence of two well-known acceleration techniques - Heavy-Ball Method and Nesterov's Accelerated Gradient on minimizing a convex, quadratic function. 
enable: true
---


---
We consider the following quadratic optimization problem:

$$ \DeclareMathOperator*{\argmin}{argmin} \newcommand{\norm}[1]{\left\lVert#1\right\rVert} \newcommand{\abs}[1]{\left\lvert#1\right\rvert} \newcommand{\bm}[1]{\boldsymbol#1} \min_{\bm x \in \mathbb{R}^n} f(\bm x) = \frac{1}{2} \bm x^T \bm A \bm x - \bm b^T \bm x + c , \tag{1} \label{equ:Abc} $$

where $\bm A \in \mathbb{R}^{n \times n}$ is a positive semi-definite matrix with eigenvalues $L = \lambda_1 \geq \ldots \geq \lambda_n = \mu > 0$. The quadratic model is a simple but powerful enough in studying local convergence of optimization algorithms. As you can see in the sequel, this nice model allows us to extract the closed-form expression of the convergence rate through basic elementary algebra. Yet before moving to the algorithms, let us make a further simplification of (\ref{equ:Abc}) by a change of variables: $\tilde{\bm x} = \bm U^T (\bm x - \bm x^\star)$, where $\bm x^\star$ is the unique minimum of the problem and $\bm U$ is the orthogonal basis in the eigenvalue decomposition of $\bm A$, i.e., $\bm A = \bm U \text{diag}(\lambda_1,\ldots,\lambda_n) \bm U^T$. Thus we have an alternative minimization

$$ \min_{\bm x \in \mathbb{R}^n} f(\bm x) = \frac{1}{2} \bm x^T \bm \Lambda \bm x \tag{2} \label{equ:standard} , $$

with $\bm \Lambda = \text{diag}(\lambda_1,\ldots,\lambda_n)$. The gradient and Hessian of the objective function are given by

$$ \nabla f(\bm x) = \bm \Lambda \bm x , \qquad \nabla^2 f(\bm x) = \bm \Lambda . $$

Furthermore, we denote $\kappa=\frac{L}{\mu} \geq 1$ the condition number of the objective function. Intuitively, the higher $\kappa$ is, the harder it is to optimize $f$. When $\kappa$ approaches $\infty$, problem (\ref{equ:standard}) is said to be *ill-conditioned*. Our goal in this note is to explain Proposition 1 in [3], given by the following table: 

<table>
	<caption>Worst-case rates for different algorithms when applied to a class of convex quadratic functions.</caption>
	<thead>
		<tr>
			<th>Method</th>
			<th>Parameter choice</th>
			<th>Convergence rate</th>
			<th>Iteration complexity</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>Gradient descent (GD)</td>
			<td>$\alpha=\frac{2}{L+\mu}$</td>
			<td>$\rho=\frac{\kappa-1}{\kappa+1}$</td>
			<td>$O(\kappa \log \frac{1}{\epsilon})$</td> 
		</tr>
		<tr>
			<td>Heavy-Ball (HB)</td>
			<td>$\alpha=\biggl(\frac{2}{\sqrt{L}+\sqrt{\mu}}\biggr)^2, \beta=\biggl( \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} \biggr)^2$</td>
			<td>$\rho=\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$</td>
			<td>$O(\sqrt{\kappa} \log \frac{1}{\epsilon})$</td> 
		</tr>
		<tr>
			<td>Nesterov's Accelerated Gradient (NAG)</td>
			<td>$\alpha=\frac{4}{3L+\mu}, \beta=\frac{\sqrt{3\kappa+1}-2}{\sqrt{3\kappa+1}+2}$</td>
			<td>$\rho=1-\frac{2}{\sqrt{3\kappa+1}}$</td>
			<td>$O(\sqrt{\kappa} \log \frac{1}{\epsilon})$</td> 
		</tr>
	</tbody>
</table>

Note that here we restrict our attention to the case with constant step sizes. Without further due let us get to the algorithmic part.


# 1. Gradient Descent

To warm-up, we consider the update of gradient descent:

$$ \bm x^{(k+1)} = \bm x^{(k)} - \alpha \nabla f(\bm x^{(k)}) \qquad \text{ for some constant } \alpha>0 . $$

Substituting the gradient of $f$ and taking the Euclidean norm, we have 

\begin{align}
\norm{\bm x^{(k+1)}} &= \norm{\bm x^{(k)} - \alpha \bm \Lambda \bm x^{(k)}} = \norm{(\bm I_n - \alpha \bm \Lambda) \bm x^{(k)}} \\\\\\
&\leq \norm{\bm I_n - \alpha \bm \Lambda}_2 \norm{\bm x^{(k)}} = \max_j \abs{1-\alpha \lambda_j} \norm{\bm x^{(k)}} \\\\\\
&= \max \Bigl\\{ \abs{1-\alpha \mu}, \abs{1-\alpha L} \Bigr\\} \norm{\bm x^{(k)}} .
\end{align}

where the last equality stems from fact that $\abs{1-\alpha \lambda}$ is a continuous and convex function of $\lambda$. Now let $\rho_\alpha = \max \Bigl\\{ \abs{1-\alpha \mu}, \abs{1-\alpha L} \Bigr\\}$, we rewrite the recursion as

$$ \norm{\bm x^{(k)}} \leq \rho_\alpha^k \norm{\bm x^{(0)}} . \tag{3} \label{equ:GD}$$

The inequality (\ref{equ:GD}) shows that GD converges linearly (or geometrically) to the optimal solution $\bm x^\star = \bm 0$ at rate $\rho_\alpha$. In order to converge, we need $\rho_\alpha < 1$. This is equivalent to having
\begin{align}
0 < \alpha < \frac{2}{L} . 
\end{align}
The optimal step size is chosen by solving
\begin{align}
\alpha_{opt} = \argmin_{0 < \alpha < \frac{2}{L}} \max \Bigl\\{ \abs{1-\alpha \mu}, \abs{1-\alpha L} \Bigr\\} . 
\end{align}
Although it can be observed that the solution happens at which the two quantities are equal, solving this minimization is non-trivial. First, it is noticeable that
\begin{align}
\abs{1-\alpha \mu} < \abs{1-\alpha L} \Leftrightarrow \alpha > \frac{2}{L+\mu} . 
\end{align}
Thus, we consider two cases:

1. If $0 < \alpha \leq \frac{2}{L+\mu}$, then
\begin{align}
\max \Bigl\\{ \abs{1-\alpha \mu}, \abs{1-\alpha L} \Bigr\\} = 1-\alpha \mu \geq 1-\frac{2\mu}{L+\mu} = \frac{L-\mu}{L+\mu} . 
\end{align}
The optimal step size in this case is $\frac{2}{L+\mu}$. Compared with the popular but *suboptimal* choice of step size we see in the literature $\alpha = \frac{1}{L}$, this step size is twice as large.

2. Otherwise, if $\frac{2}{L+\mu} < \alpha < \frac{2}{L}$, then
\begin{align}
\max \Bigl\\{ \abs{1-\alpha \mu}, \abs{1-\alpha L} \Bigr\\} = \alpha L - 1 > \frac{2L}{L+\mu} - 1 = \frac{L-\mu}{L+\mu} . 
\end{align}

In summary, we have $\alpha_{opt}=\frac{2}{L+\mu}$ and $\rho_{opt}=\frac{\kappa-1}{\kappa+1}$. At this optimal step size, the number of iterations to reach (relative) $\epsilon$-accuracy, i.e., $\frac{\norm{\bm x^{(k)}}}{\norm{\bm x^{(0)}}} < \epsilon \ll 1$, satisfies 
\begin{align}
\rho_{opt}^k < \epsilon \Rightarrow k > \frac{\log 1/\epsilon}{\log \bigl(1 - \frac{2}{\kappa+1} \bigr)} \approx \frac{\kappa+1}{2} \log \frac{1}{\epsilon} . 
\end{align}
In other words, the iteration complexity of GD with optimal step size is $O(\kappa \log \frac{1}{\epsilon})$.



# 2. Heavy-Ball Method

The Heavy-Ball method adds a second term from the previous iterate to the update equation of gradient descent:

$$ \bm x^{(k+1)} = \bm x^{(k)} - \alpha \nabla f(\bm x^{(k)}) + \beta (\bm x^{(k)} - \bm x^{(k-1)}) \qquad \text{ for some constants } \alpha>0, \beta\geq 0 . $$

On a standard quadratic, the update can be further simplified as

$$ \bm x^{(k+1)} = \bm x^{(k)} - \alpha \bm \Lambda \bm x^{(k)} + \beta (\bm x^{(k)} - \bm x^{(k-1)}) = \bigl( (1+\beta) \bm I_n - \alpha \bm \Lambda \bigr) \bm x^{(k)} - \beta \bm x^{(k-1)} . $$

Stacking the current iterate and the previous one and taking the norm yield
\begin{align}
\begin{bmatrix} \bm x^{(k+1)} \\\\\\ \bm x^{(k)} \end{bmatrix} = \begin{bmatrix} (1+\beta)\bm I_n - \alpha \bm \Lambda & -\beta \bm I_n \\\\\\ \bm I_n & \bm 0 \end{bmatrix} \begin{bmatrix} \bm x^{(k)} \\\\\\ \bm x^{(k-1)} \end{bmatrix} . 
\end{align}
Denote $\bm y^{(k)} = \begin{bmatrix} \bm x^{(k+1)} \\\\\\ \bm x^{(k)} \end{bmatrix}$ and $\bm T = \begin{bmatrix} (1+\beta)\bm I_n - \alpha \bm \Lambda & -\beta \\\\\\ \bm I_n & \bm 0 \end{bmatrix}$, we derive the exponential decrease in the norm of $\bm y$:

$$ \norm{\bm y^{(k)}} = \norm{\bm T \bm y^{(k-1)}} = \norm{\bm T^k \bm y^{(0)}} \leq \norm{\bm T^k}_2 \norm{\bm y^{(0)}} \leq \bigl(\rho(\bm T) + o(1) \bigr)^k \norm{\bm y^{(0)}} , \tag{4} \label{equ:HB} $$

where $\rho(\bm T)$ is the [spectral radius](https://en.wikipedia.org/wiki/Spectral_radius) of $\bm T$ and the last inequality uses [Gelfand’s formula](https://en.wikipedia.org/wiki/Spectral_radius#Gelfand's_formula). Thus, in order to determine the convergence rate, we will need to find the eigenvalues of $\bm T$ and determine their maximum absolute value. By carefully looking at its special structure, one can show that $\bm T$ is permutation-similar to a block diagonal matrix:
\begin{align}
\bm T \sim \begin{bmatrix} \bm T_1 & \bm 0 & \ldots & \bm 0 \\\\\\ \bm 0 & \bm T_2 & \ldots & \bm 0 \\\\\\ \vdots & & \ldots & \vdots \\\\\\ \bm 0 & \bm 0 & \ldots & \bm T_n \end{bmatrix} \quad \text{where} \quad \bm T_j = \begin{bmatrix} 1+\beta-\alpha \lambda_j & -\beta \\\\\\ 1 & 0 \end{bmatrix} \text{ for } j=1,2,\ldots,n . 
\end{align}
Hence, the eigenvalues of $\bm T$ are the union of the eigenvalues of $\bm T_j$. For each $j$, the eigenvalues of $\bm T_j$ are the two roots of the equation

$$ r^2 - (1+\beta - \alpha \lambda_j) r + \beta = 0 \quad \text{with} \quad \Delta_j = (1+\beta - \alpha \lambda_j)^2 - 4\beta . $$

Therefore, we can bound the convergence rate by
\begin{align}
\rho_{\alpha,\beta} \approx \rho(\bm T) = \max_j r_j = \max \{r_1, r_n\}, \quad \text{where } r_j = \begin{cases} \frac{1}{2} \Bigl( \abs{1+\beta-\alpha \lambda_j} + \sqrt{\Delta_j} \Bigr) & \text{if } \Delta_j > 0, \\\\\\ \sqrt{\beta} & \text{otherwise.} \end{cases} 
\end{align}
Here we can think of $r$ as a continuous and quasiconvex function of $\lambda$. The choice of step sizes must satisfy $\rho_{\alpha,\beta} < 1$, or equivalently,
\begin{align}
\begin{cases} 0 \leq \beta < 1, \\\\\\ 0 < \alpha < \frac{2(1+\beta)}{L} . \tag{5} \label{equ:HB_range} \end{cases} 
\end{align}
The optimal step size is chosen by minimizing $\rho_{\alpha,\beta}$ over the aforementioned range. This optimization is more complicated than the one with GD. By observing that 
\begin{align}
\Delta_j \leq 0 \Leftrightarrow \beta \geq \bigl(1-\sqrt{\alpha \lambda_j}\bigr)^2 \quad \text{and} \quad \abs{1-\sqrt{\alpha \mu}} < \abs{1-\sqrt{\alpha L}} \Leftrightarrow \alpha > \biggl( \frac{2}{\sqrt{L}+\sqrt{\mu}} \biggr)^2 , 
\end{align}
we can break it into $4$ cases:

1. If $0 < \alpha \leq \Bigl( \frac{2}{\sqrt{L}+\sqrt{\mu}} \Bigr)^2$ and $\beta \geq (1-\sqrt{\alpha \mu})^2$, then
\begin{align}
\max \\{ r_1, r_n \\} = \sqrt{\beta} \geq 1-\sqrt{\alpha \mu} \geq 1 - \frac{2\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}} = \frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}} . 
\end{align}
In this case, the equality holds at the boundary.

2. If $0 < \alpha \leq \Bigl( \frac{2}{\sqrt{L}+\sqrt{\mu}} \Bigr)^2$ and $\beta < (1-\sqrt{\alpha \mu})^2$, then
\begin{align}
\max \\{ r_1, r_n \\} \geq r_n(\alpha,\beta) = \frac{1}{2} \Bigl( 1+\beta-\alpha \mu + \sqrt{(1+\beta - \alpha \mu)^2 - 4\beta} \Bigr) . 
\end{align}
It can be verified that $r_n(\alpha,\beta)$ is a decreasing function of $\beta$. Hence, 
\begin{align}
\max \\{ r_1, r_n \\} > r_n\bigl(\alpha,(1-\sqrt{\alpha \mu})^2\bigr) = 1-\sqrt{\alpha \mu} \geq \frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}} . 
\end{align}

3. If $\alpha > \Bigl( \frac{2}{\sqrt{L}+\sqrt{\mu}} \Bigr)^2$ and $\beta \geq (\sqrt{\alpha L}-1)^2$, then
\begin{align}
\max \\{ r_1, r_n \\} = \sqrt{\beta} \geq \sqrt{\alpha L}-1 > \frac{2\sqrt{L}}{\sqrt{L}+\sqrt{\mu}}-1 = \frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}} .
\end{align}

4. If $\alpha > \Bigl( \frac{2}{\sqrt{L}+\sqrt{\mu}} \Bigr)^2$ and $\beta < (\sqrt{\alpha L}-1)^2$, then
\begin{align}
\max \\{ r_1, r_n \\} \geq r_1(\alpha,\beta) = \frac{1}{2} \Bigl( -1-\beta+\alpha \mu + \sqrt{(1+\beta - \alpha \mu)^2 - 4\beta} \Bigr) .
\end{align}
Similarly, $r_1(\alpha,\beta)$ is a decreasing function of $\beta$ and 
\begin{align}
\max \\{ r_1, r_n \\} > r_1\bigl(\alpha,(\sqrt{\alpha L}-1)^2\bigr) = \sqrt{\alpha L}-1 \geq \frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}} . 
\end{align}

In summary, we have 

$$\alpha_{opt}=\biggl( \frac{2}{\sqrt{L}+\sqrt{\mu}} \biggr)^2, \quad \beta_{opt} = \biggl( \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} \biggr)^2 \quad \text{and} \quad \rho_{opt}=\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} . \tag{6} \label{equ:HB_optimal} $$

With this optimal choice, the number of iterations to reach (relative) $\epsilon$-accuracy satisfies 
\begin{align}
\rho_{opt}^k < \epsilon \Rightarrow k > \frac{\log 1/\epsilon}{\log \bigl(1 - \frac{2}{\sqrt{\kappa}+1} \bigr)} \approx \frac{\sqrt{\kappa}+1}{2} \log \frac{1}{\epsilon} . 
\end{align}
In other words, the iteration complexity of HB with optimal step size is $O(\sqrt{\kappa} \log \frac{1}{\epsilon})$.


# 3. Nesterov's Accelerated Gradient

The HB update can be rewritten as

$$ \begin{cases} \bm y^{(k+1)} = \bm x^{(k)} - \alpha \nabla f(\bm x^{(k)}) , \\
\bm x^{(k+1)} = \bm y^{(k+1)} + \beta (\bm x^{(k)} - \bm x^{(k-1)}) \end{cases} . $$

While Polyak's method evaluates the gradient before adding momentum, Nesterov suggested the other way around: evaluates the gradient after applying momentum

$$ \begin{cases} \bm y^{(k+1)} = \bm x^{(k)} - \alpha \nabla f(\bm x^{(k)}) , \\
\bm x^{(k+1)} = \bm y^{(k+1)} + \beta (\bm y^{(k)} - \bm y^{(k-1)}) \end{cases} . $$

This simple change produces a movement through the "lookahead" gradient that potentially corrects for future oscillations. The Nesterov's accelerated gradient update can be represented in one line as

$$ \bm x^{(k+1)} = \bm x^{(k)} + \beta (\bm x^{(k)} - \bm x^{(k-1)}) - \alpha \nabla f \bigl( \bm x^{(k)} + \beta (\bm x^{(k)} - \bm x^{(k-1)}) \bigr) . $$

Substituting the gradient of $f$ in quadratic case yields

\begin{align} 
\bm x^{(k+1)} &= \bm x^{(k)} + \beta (\bm x^{(k)} - \bm x^{(k-1)}) - \alpha \bm \Lambda \bigl( \bm x^{(k)} + \beta (\bm x^{(k)} - \bm x^{(k-1)}) \bigr) \\\\\\
&= (1+\beta) (\bm I_n - \alpha \bm \Lambda) \bm x^{(k)} - \beta (\bm I_n - \alpha \bm \Lambda) \bm x^{(k-1)} . 
\end{align}

Now one could attempt to derive a similar expression to (\ref{equ:HB}), but there is a better way to tackle the problem that gets rid of the $o(1)$ term in the convergence rate. By breaking the recursion into each dimension, we obtain the corresponding second order linear difference equation

$$ x_j^{(k+1)} - (1+\beta)(1-\alpha \lambda_j) x_j^{(k)} + \beta (1-\alpha \lambda_j) x_j^{(k-1)} = 0 . $$

Solving the characteristic equation 

$$ r^2 - (1+\beta)(1-\alpha \lambda_j) r + \beta (1-\alpha \lambda_j) = 0 , \quad \text{with} \quad \Delta_j = (1+\beta)^2 (1-\alpha \lambda_j)^2 - 4 \beta (1-\alpha \lambda_j) ,$$

yields three cases:

1. If $\Delta_j = 0$, then there are two repeated real roots $r_{12} = \frac{1}{2} (1+\beta)(1-\alpha \lambda_j)$ and 
\begin{align}
x_j^{(k)} = (C_1 + C_2 k) r_{12}^k \quad \text{for some constants } C_1, C_2 \text{ depend on the initial point} .
\end{align}

2. If $\Delta_j > 0$, then there are two distinct real roots $r_{12} = \frac{1}{2} \Bigl( (1+\beta)(1-\alpha \lambda_j) \pm \sqrt{\Delta_j} \Bigr)$ and 
\begin{align}
x_j^{(k)} = C_1 r_{1}^k + C_2 r_{2}^k \quad \text{for some constants } C_1, C_2 \text{ depend on the initial point}.
\end{align}

3. If $\Delta_j < 0$, then there are two conjugate complex roots satisfying $\abs{r_{12}} = \sqrt{\beta (1-\alpha \lambda_j)}$ and 
\begin{align}
x_j^{(k)} = \Bigl( \sqrt{\beta (1-\alpha \lambda_j)} \Bigr)^k \bigl( C_1 \cos(\theta k) + C_2 \sin(\theta k) \bigr) ,
\end{align}
for $\cos \theta = \frac{1+\beta}{2} \sqrt{\frac{1-\alpha \lambda_j}{\beta}}$ and some constants $C_1$, $C_2$ depend on the initial point.

In all cases, we can bound the convergence rate by
\begin{align}
\rho_{\alpha,\beta} = \rho(\bm T) = \max_j r_j = \max \{r_1, r_n\}, \quad \text{where } r_j = \begin{cases} \frac{1}{2} \Bigl( (1+\beta)\abs{1-\alpha \lambda_j} + \sqrt{\Delta_j} \Bigr) & \text{if } \Delta_j > 0, \\\\\\ \sqrt{\beta \abs{1-\alpha \lambda_j}} & \text{otherwise.} \end{cases}
\end{align}
In order to converge, we need 
\begin{align}
\rho_{\alpha,\beta} < 1 \quad \Leftrightarrow \quad \beta>0 \text{ and } \max \Bigl\\{ 0, \frac{\beta-1}{\mu \beta} \Bigr\\} < \alpha < \frac{2(\beta+1)}{L(2\beta+1)} . \tag{7} \label{equ:NAG_range} 
\end{align}

Choosing the optimal step size in this range can be proceeded as follows. Let us first rewrite $\Delta_j$ as

$$ \Delta_j = (1-\alpha \lambda)^2 \biggl( \beta - \frac{1-\sqrt{\alpha \lambda_j}}{1+\sqrt{\alpha \lambda_j}} \biggr) \biggl( \beta - \frac{1+\sqrt{\alpha \lambda_j}}{1-\sqrt{\alpha \lambda_j}} \biggr) . $$

Thus, if $\alpha \lambda_j > 1$, $r_j=\frac{1}{2} \Bigl( (1+\beta)(\alpha \lambda_j-1) + \sqrt{\Delta_j} \Bigr)$ is an increasing function of $\beta$. Otherwise, $r_j(\beta)$ is decreasing in the interval $0<\beta<\frac{1-\sqrt{\alpha \lambda_j}}{1+\sqrt{\alpha \lambda_j}}$ and increasing in the interval $\beta>\frac{1-\sqrt{\alpha \lambda_j}}{1+\sqrt{\alpha \lambda_j}}$, with the minimum at $r_j\Bigl(\frac{1-\sqrt{\alpha \lambda_j}}{1+\sqrt{\alpha \lambda_j}}\Bigr) = 1-\sqrt{\alpha \lambda_j}$. Thus, we can break it into $3$ cases:

1. If $\alpha \leq \frac{1}{L}$, then 
\begin{align}
\max \\{ r_1, r_n \\} \geq r_n \geq r_n \biggl( \frac{1-\sqrt{\alpha \mu}}{1+\sqrt{\alpha \mu}} \biggr) = 1-\sqrt{\alpha \mu} \geq 1-\sqrt{\frac{\mu}{L}} .
\end{align}
In this case, the equality holds at $\alpha = \frac{1}{L}$ and $\beta=\frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}}$. However, the rate $\rho = 1-\sqrt{\frac{\mu}{L}}$ is **suboptimal**.

2. If $\frac{1}{L} < \alpha \leq \frac{1}{\mu}$, then
	1. If $\frac{1-\sqrt{\alpha \mu}}{1+\sqrt{\alpha \mu}} \leq \beta \leq \frac{1+\sqrt{\alpha \mu}}{1-\sqrt{\alpha \mu}}$, we have
	\begin{align}
	\max \\{ r_1, r_n \\} = \max \biggl\\{ \frac{1}{2} \Bigl( (1+\beta)(\alpha L-1) + \sqrt{\Delta_1} \Bigr) , \sqrt{\beta(1-\alpha \mu)} \biggr\\} .
	\end{align}
	Notice that both terms inside $\max$ are increasing function of $\beta$. However, the former term is an increasing function of $\alpha$ while the later is a decreasing function of $\alpha$. Thus, minimizing over $\alpha$ and $\beta$ in the aforementioned range yields the solution where the two terms are equal **and** $\beta$ is at its minimum value, i.e., 
	\begin{align}
		\beta=\frac{1-\sqrt{\alpha \mu}}{1+\sqrt{\alpha \mu}} \quad \text{and} \quad \frac{1}{2} \Bigl( (1+\beta)(\alpha L-1) + \sqrt{\Delta_1} \Bigr) = \sqrt{\beta(1-\alpha \mu)} .
	\end{align}
	Solving this yields $\alpha = \frac{4}{3L+\mu}$ and $\beta=\frac{\sqrt{3L+\mu}-2\sqrt{\mu}}{\sqrt{3L+\mu}+2\sqrt{\mu}}$, with the **optimal** rate
	\begin{align}
		\rho = 1-\frac{2\sqrt{\mu}}{\sqrt{3L+\mu}} \leq 1-\sqrt{\frac{\mu}{L}} .
	\end{align}
	2. If $\beta < \frac{1-\sqrt{\alpha \mu}}{1+\sqrt{\alpha \mu}}$, we have
	\begin{align}
		\max \\{ r_1, r_n \\} = \max \biggl\\{ \frac{1}{2} \Bigl( (1+\beta)(\alpha L-1) + \sqrt{\Delta_1} \Bigr) , \frac{1}{2} \Bigl( (1+\beta)(1-\alpha \mu) + \sqrt{\Delta_n} \Bigr) \biggr\\} .
	\end{align}
	We need to show that the maximum is always greater than $1-\frac{2\sqrt{\mu}}{\sqrt{3L+\mu}}$, <span style="color: red;">which is not straightforward to me though!</span>
	3. The case $\beta > \frac{1+\sqrt{\alpha \mu}}{1-\sqrt{\alpha \mu}}$ does not converge since
	\begin{align}
	\max \\{ r_1, r_n \\} \geq \sqrt{\beta (1-\alpha \mu)} > \frac{1+\sqrt{\alpha \mu}}{1-\sqrt{\alpha \mu}} (1-\alpha \mu) > 1 .
	\end{align}

3. If $\alpha > \frac{1}{\mu}$, then
\begin{align}
\max \\{ r_1, r_n \\} = r_1 = \frac{1}{2} \Bigl( (1+\beta)(\alpha L-1) + \sqrt{\Delta_1} \Bigr) \geq \alpha L - 1 > \frac{L}{\mu} - 1 \geq 1-\frac{\mu}{L} \geq 1-\sqrt{\frac{\mu}{L}} .
\end{align}

To conclude, the optimal choice of step sizes is given by

$$ \alpha_{opt}=\frac{4}{3L+\mu}, \quad \beta_{opt}=\frac{\sqrt{3\kappa+1}-2}{\sqrt{3\kappa+1}+2}, \quad \rho_{opt}=1-\frac{2}{\sqrt{3\kappa+1}} . \tag{8} \label{equ:NAG_optimal} $$

With this optimal choice, the number of iterations to reach (relative) $\epsilon$-accuracy satisfies
\begin{align}
\rho_{opt}^k < \epsilon \Rightarrow k > \frac{\log 1/\epsilon}{\log \bigl( 1-\frac{2}{\sqrt{3\kappa+1}} \bigr)} \approx \frac{\sqrt{3\kappa+1}}{2} \log \frac{1}{\epsilon} . 
\end{align}
The iteration complexity of NAG with optimal step size is the same as HB - $O(\sqrt{\kappa} \log \frac{1}{\epsilon})$.

> As a final note, HB generally does not converge when the objective is non-quadratic, strongly convex and smooth. On the other hand, NAG can be guaranteed to converge as long as the step sizes are chosen properly (see [2]). In the next note, we will study the proof of convergence of NAG on non-quadratic case.


## References
> 1. B. T. Polyak, Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17, 1964. 
> 2. Y. Nesterov, Introductory lectures on convex optimization: a basic course, Kluwer Academic Publishers, 2004.
> 3. L. Lessard, B. Recht, & A. Packard, Analysis and design of optimization algorithms via integral quadratic constraints. SIAM Journal on Optimization, 26(1), 57–95, 2016.

