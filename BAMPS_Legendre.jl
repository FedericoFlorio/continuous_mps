using TensorTrains, LogarithmicNumbers, Tullio, Polynomials, SpecialPolynomials
using Lazy: @forward

potts2spin(x,q) = (x-1)/(q-1)*2 - 1
spin2potts(σ,q) = (σ+1)/2*(q-1) + 1

reduce_domain(σ, dᵢ) = σ/dᵢ

# Here the basis functions are Legendre polynomials
u(α::Int) = Legendre(vcat(zeros(α), sqrt(α+1/2)))

function check_bond_dims(tensors::Vector{Array{F,N}}) where {F<:Number, N}
    T = length(tensors)
    for t in 1:lastindex(tensors)
        dᵗ = size(tensors[t],2)
        dᵗ⁺¹ = size(tensors[mod1(t+1,T)], 1)
        if dᵗ != dᵗ⁺¹
            println("Wrong bond size for matrix t=$t. dᵗ=$dᵗ, dᵗ⁺¹=$dᵗ⁺¹")
            return false
        end
    end
    return true
end

mutable struct BasisTensorTrain{F<:Number,N} <: AbstractTensorTrain{F,N}
    tensors::Vector{Array{F,N}}
    z::Logarithmic{F}

    function BasisTensorTrain{F,N}(tensors::Vector{Array{F,N}}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Number, N}
        N > 1 || throw(ArgumentError("Tensors shold have at least 3 indices"))

        size(tensors[1],1) == 1 ||
            throw(ArgumentError("First matrice must have 1 row"))
        size(tensors[end],2) == 1 ||
            throw(ArgumentError("Last matrices must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        return new{F,N}(tensors, z)
    end
end
function BasisTensorTrain(tensors::Vector{Array{F,N}}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Number, N} 
    return BasisTensorTrain{F,N}(tensors; z)
end

@forward BasisTensorTrain.tensors Base.getindex, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.length, Base.eachindex,  
    check_bond_dims


function BAMPS(b::TensorTrain, K::Int, d::Int; eps=1e-1)
    # Finds the scalar product between the Legendre polynomial of degree l and a train of deltas with centres xs and areas As
    function fitLegendre(l, xs, As)
        length(xs) == length(As) || throw(ArgumentError("There must be the same numbers of centres and areas"))
        aₗ = sum([As[i]/eps * integrate(u(l), xs[i]-eps/2, xs[i]+eps/2) for i in eachindex(xs)])
    end

    G = [zeros(size(bᵗ)[1:end-1]..., K) for bᵗ in b]
    for (t,Aᵗ) in enumerate(b)
        q = size(Aᵗ)[end]
        Gᵗ = G[t]
        @tullio Gᵗ[m,n,α] = fitLegendre(α-1, reduce_domain.(potts2spin.(1:q,q),d), Aᵗ[m,n,:])
    end
    return G |> BasisTensorTrain
end

function reconstruct_BAMPS(bBasis::BasisTensorTrain, q::Int, d::Int)
    T = length(bBasis)
    K = size(bBasis[1])[end]
    F = eltype(bBasis[1])

    U = [u(α-1)(reduce_domain(potts2spin(y,q), d)) for α in 1:K, y in 1:q]
    b = Array{Array{F, 3},1}(undef,T)

    for t in 1:T
        bᵗ = b[t]
        bBasisᵗ = bBasis[t]
        @tullio bᵗ[m,n,y] = bBasisᵗ[m,n,α]*U[α,y]
    end

    return b |> TensorTrain
end



;