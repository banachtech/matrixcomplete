# Matrix Completion

Consider a $m \times n$ matrix $X$ with missing entries indexed by the set $\Omega = \{(i,j) : i \in \{1,\ldots,m\}, j \in \{1, \ldots, n\}, X_{ij} \text{is missing}\}$. The problem is to find a low-rank approximation of $X$ that is close to $X_{ij}$ for $(i,j)$ corresponding to known values of $X$.

$$ \text{Minimize} \quad rank(M) $$

$$ \text{s.t.} \quad \sum_{i,j \in \Omega} (X_{ij} - M_{ij})^2 \leq \delta $$


One way is to use a low rank factorization of $M$ to solve:

$$ \underset {A, B}{Minimize} \quad \frac{1}{2}\Vert P_\Omega(X - A B^T) \Vert_F^2 + \frac{\lambda}{2} \left(\Vert A \Vert_F^2 + \Vert B \Vert_F^2\right),$$

where $A$ is $m \times r$, $B$ is $r \times n$ and $r < min(m,n)$. While non-convex in $A$ and $B$, this is bi-convex. With $A$ fixed, solving for $B$ is equivalent to solving series of ridge regressions and vice-versa. 

Alternating least squares algorithm ("SOFT-IMPUTE ALS") described below solves the above problem.

$$ k \leftarrow k+1 $$

$$ X^* \leftarrow P_\Omega(X) + P_\Omega^\perp(A B^T) $$

$$ A \leftarrow X^* B (B^T B+\lambda I)^{-1} $$

$$ X^* \leftarrow P_\Omega(X) + P_\Omega^\perp(A\;B^T) $$

$$ B \leftarrow (X^*)^T A(A^T A+\lambda I)^{-1} $$

