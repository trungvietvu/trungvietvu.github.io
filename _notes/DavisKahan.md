---
title: "Matrix Perturbation and Davis-Kahan Theorem"
collection: notes
permalink: /notes/2020/DavisKahan
date: 2020-06-20
excerpt: In this note we will study what happens when adding small perturbations to a symmetric matrix. How much can the invariant subspace spanned by its eigenvectors change?  
enable: true
---


---
# 1. Eigensubspace decomposition and matrix perturbation

Let $\DeclareMathOperator*{\argmin}{argmin} \newcommand{\norm}[1]{\left\lVert#1\right\rVert} \newcommand{\bm}[1]{\boldsymbol#1} \bm M \in \mathbb{R}^{n \times n}$ be a symmetric matrix with eigenvalue decomposition $\bm M=\bm U \bm \Lambda \bm U^T$. Here $\bm U \in \mathbb{R}^{n \times n}$ is an orthogonal matrix, i.e., $\bm U^T \bm U = \bm U \bm U^T = \bm I_n$, and $\bm \Lambda=\text{diag}(\lambda_1,\lambda_2,\ldots,\lambda_n)$ has the eigenvalues of $\bm M$ as its diagonal entries. Given an integer $r<n$, we can define the eigensubspace decomposition as the partition

$$ \bm M = \begin{bmatrix} \bm U_1 & \bm U_2 \end{bmatrix} \begin{bmatrix} \bm \Lambda_1 & 0 \\ 0 & \bm \Lambda_2 \end{bmatrix} \begin{bmatrix} \bm U_1^T \\ \bm U_2^T \end{bmatrix}, \quad \text{ where } \bm \Lambda_1 = \text{diag}(\lambda_1,\ldots,\lambda_r) , \bm \Lambda_2 = \text{diag}(\lambda_{r+1},\ldots,\lambda_n) . $$

The submatrices $\bm U_1$ and $\bm U_2$ are called *semi-orthogonal* since $\bm U_1^T \bm U_1 = \bm I_r$ and $\bm U_2^T \bm U_2 = \bm I_{n-r}$. The orthogonal projections onto the subspaces spanned by the columns of $\bm U_1$ and $\bm U_2$ are $\bm P_{\bm U_1} = \bm U_1 \bm U_1^T$ and $\bm P_{\bm U_2} = \bm U_2 \bm U_2^T$, respectively. Since $\bm P_{\bm U_1} + \bm P_{\bm U_1} = \bm I_n$, the two corresponding subspaces are orthogonal complement.

Now let $\bm \Delta \in \mathbb{R}^{n \times n}$ be a small perturbation (symmetric), we are interested in how the eigenvalues and eigenvectors of $\bm M$ change. Denote the eigensubspace decomposition of $\tilde{\bm M} = \bm M + \bm \Delta$ by

$$ \tilde{\bm M} = \tilde{\bm U} \tilde{\bm \Lambda} \tilde{\bm U}^T = \begin{bmatrix} \tilde{\bm U}_1 & \tilde{\bm U}_2 \end{bmatrix} \begin{bmatrix} \tilde{\bm \Lambda}_1 & 0 \\ 0 & \tilde{\bm \Lambda}_2 \end{bmatrix} \begin{bmatrix} \tilde{\bm U}_1^T \\ \tilde{\bm U}_2^T \end{bmatrix} . $$ 

The Weyl's theorem tells us that the eigenvalues are fairly stable under small perturbations:

Theorem 1. 
: $\norm{\tilde{\bm \Lambda} - \bm \Lambda}_2 \leq \norm{\bm \Delta}_2 .$


**Proof.**

We will prove a more general result: let $\bm A, \bm B \in \mathbb{R}^{n \times n}$ be symmetric and $\lambda_i(.)$ be the *i*th largest eigenvalue of a matrix, then

$$ \lambda_i(\bm A) + \lambda_j(\bm B) \geq \lambda_{i+j-1} (\bm A + \bm B) . $$

In order to prove the above inequality, we use the min-max theorem as follows:

$$ \lambda_i(\bm A) = \min_{\mathcal{V}_A: \text{dim}(\mathcal{V}_A)=n-i+1} \max_{\bm x \in \mathcal{V}_A, \norm{\bm x}=1} \bm x^T \bm A \bm x \quad \text{ and } \quad \lambda_j(\bm B) = \min_{\mathcal{V}_B: \text{dim}(\mathcal{V}_B)=n-j+1} \max_{\bm x \in \mathcal{V}_B, \norm{\bm x}=1} \bm x^T \bm B \bm x .$$

Let 

$$\mathcal{V}_A^* = \argmin_{\mathcal{V}_A: \text{dim}(\mathcal{V}_A)=n-i+1}\max_{\bm x \in \mathcal{V}_A, \norm{\bm x}=1} \bm x^T \bm A \bm x \quad \text{ and } \quad \mathcal{V}_B^* = \arg\min_{\mathcal{V}_B: \text{dim}(\mathcal{V}_B)=n-j+1} \max_{\bm x \in \mathcal{V}_B, \norm{\bm x}=1} \bm x^T \bm B \bm x . $$


Denote $\mathcal{V} = \mathcal{V}_A^* \cap \mathcal{V}_B^*$, where $\text{dim}(\mathcal{V}) \geq (n-i+1)+(n-j+1)-n = n+2-i-j$. We have

$$\begin{aligned} \lambda_i(\bm A) + \lambda_j(\bm B) &\geq \max_{\bm x \in \mathcal{V}, \norm{\bm x}=1} \bm x^T (\bm A + \bm B) \bm x \\
&\geq \min_{\mathcal{V}_{A+B}: \text{dim}(\mathcal{V}_A)=n+2-i-j} \max_{\bm x \in \mathcal{V}_A, \norm{\bm x}=1} \bm x^T (\bm A+\bm B) \bm x = \lambda_{i+j-1} (\bm A + \bm B) . \qquad \blacksquare \end{aligned}$$




# 2. Principle angles between two subspaces



# Orthogonal Procrustes theorem


# The $\sin \theta$ theorem


## References
> 1. C. Davis and W.M. Kahan. The rotation of eigenvectors by a perturbation. III. SIAM Journal on Numerical Analysis, 7(1), pp.1-46, 1970.
> 2. P.Å., Wedin. Perturbation bounds in connection with singular value decomposition. BIT Numerical Mathematics, 12(1), pp.99-111, 1972.





