module MatrixComplete

using LinearAlgebra, Random, PyPlot

export project, createTestMatrix, softImputeALS, printMatrix, randOrtho, isFilled

function project(X, Ω)
    Y = deepcopy(X)
    Y[Ω] .= 0.0
    return Y
end

function createTestMatrix(m, n, p)
    A = rand([missing; randn(p)], m, n)
    return A
end

function randOrtho(m,n)
    p = max(m,n)
    Q, R = householder(randn(p,p))
    O = Q * Diagonal(sign.(diag(R)))
    return O[1:m, 1:n]
end

function printMatrix(A; marker = ".", markersize = "1")
    spy(A, marker = marker, markersize = markersize)
end

function softImputeALS(X, λ, r; tol = 1e-4, iter = 1000)
    Y = deepcopy(X)
    m,n = size(Y)
    Ω = findall(!isFilled, Y)
    Ψ = findall(isFilled,Y)
    A0 = randOrtho(m, r)
    B0 = randOrtho(n, r)
    i = 0
    while true
        Y = project(Y, Ω) + project(A0 * B0', Ψ)
        A = Y * B0 * inv(B0' * B0 + λ * I)
        Y = project(Y, Ω) + project(A * B0', Ψ)
        B = Y' * A * inv(A' * A + λ * I)
        δ = norm(A * B' - A0 * B0', 2) / norm(A0 * B0', 2)
        if δ < tol || i > iter
            break
        end
        A0 = A
        B0 = B
        i = i + 1
    end
    return A0, B0
end

function householder(A)
    m,n = size(A)
    Q = diagm(ones(m))
    t = min(m-1, n)
    R = A
    for j = 1:t
        QQ = direct_sum(getQ(R[j:m, j:n]), m)
        Q = QQ * Q
        R = QQ*R
    end
    return Q', R
end

function getQ(X)
    m = size(X, 1)
    a = X[:,1]
    v = a - norm(a) * (1:m .== 1)
    v = v / norm(v)
    return diagm(ones(m)) - 2 * v * v'
end

function direct_sum(X, m)
    P = diagm(ones(m))
    n = m - size(X, 1) + 1
    P[n:m, n:m] = X
    return P
end

function isFilled(x)
    if ismissing(x) | isnan(x)
        return false
    else
        return true
    end
end

end
