using LegendrePolynomials

function LegendreCoeff(n::Int, k::Int ; norm=Val(:standard))
    0 ≤ k ≤ n || throw(ArgumentError("It must be 0 ≤ k ≤ n"))
    aⁿₖ = 2^n * binomial(n,k) * binomial((n+k-1)/2,n)
    return maybenormalize(aⁿₖ, n, norm)
end