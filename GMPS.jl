using TensorTrains
using Lazy: @forward
using MatrixProductBP: MPEM1

potts2spin(x) = 3-2x
spin2potts(σ) = (3-σ)/2

function check_bond_dims(tensors::Vector{Vector{Array{F,N}}}) where {F<:Number, N}
    for t in 1:lastindex(tensors[1])
        dᵗ = size(tensors[1][t],2)
        dᵗ⁺¹ = size(tensors[1][mod1(t+1, length(tensors[1]))],1)
        if dᵗ != dᵗ⁺¹
            println("Bond size for matrix t=$t. dᵗ=$dᵗ, dᵗ⁺¹=$dᵗ⁺¹")
            return false
        end
        if any(!=(dᵗ), [size(tensors[α][t],2) for α in 2:lastindex(tensors)])
            println("Bond sizes for matrices t=$t")
            return false
        end
    end
    return true
end

mutable struct GaussianTensorTrain{F<:Number,N} <: AbstractTensorTrain{F,N}
    tensors::Vector{Vector{Array{F,N}}}
    z::Logarithmic{F}

    function GaussianTensorTrain{F,N}(tensors::Vector{Vector{Array{F,N}}}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Number, N}
        N > 1 || throw(ArgumentError("Tensors shold have at least 2 indices"))
        size(tensors[1][1],1) == size(tensors[2][1],1) == size(tensors[3][1],1) == 1 ||
            throw(ArgumentError("First matrices must have 1 row"))
        size(tensors[1][end],2) == size(tensors[2][end],2) == size(tensors[3][end],2) == 1||
        throw(ArgumentError("Last matrices must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        return new{F,N}(tensors, z)
    end
end
function GaussianTensorTrain(tensors::Vector{Vector{Array{F,N}}}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Number, N} 
    return GaussianTensorTrain{F,N}(tensors; z)
end

@forward GaussianTensorTrain.tensors Base.getindex, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.length, Base.eachindex,  
    check_bond_dims


function GMPS(b::TensorTrain)
    G = [[zeros(size(bᵗ)[1:end-1]) for bᵗ in b] for α in 1:3]
    for (t,Aᵗ) in enumerate(b)
        for α in 1:3
            G[α][t] = [sum([potts2spin(y)^(α-1)*Aᵗ[m,n,y] for y in axes(Aᵗ)[end]]) for m in axes(Aᵗ)[1], n in axes(Aᵗ)[2]]
        end
    end
    return GaussianTensorTrain(G)
end

function reconstruct_GMPS(bGauss::GaussianTensorTrain)
    T = lastindex(bGauss[1])
    F = eltype(bGauss[1][1])
    b = Array{Array{F, 3},1}(undef,T)

    for t in eachindex(bGauss[1])
        Bᵗ₀, Bᵗ₁, Bᵗ₂ = bGauss[1][t], bGauss[2][t], bGauss[3][t]
        Aᵗ = [Bᵗ₀[m,n]^2 / sqrt(2*π*(Bᵗ₂[m,n]*Bᵗ₀[m,n] - Bᵗ₁[m,n]^2)) * exp(-0.5*(potts2spin(y)*Bᵗ₀[m,n] - Bᵗ₁[m,n])^2/(Bᵗ₂[m,n]*Bᵗ₀[m,n] - Bᵗ₁[m,n]^2)) for m in axes(Bᵗ₀)[1], n in axes(Bᵗ₀)[2], y in 1:2]
        # Aᵗ = [Bᵗ₀[m,n]!=0 ? sign(Bᵗ₀[m,n])Bᵗ₀[m,n]^2 / sqrt(2*π*(abs(Bᵗ₂[m,n]*Bᵗ₀[m,n] - Bᵗ₁[m,n]^2))) * exp(-0.5*(potts2spin(y)*Bᵗ₀[m,n] - Bᵗ₁[m,n])^2/(Bᵗ₂[m,n]*Bᵗ₀[m,n] - Bᵗ₁[m,n]^2)) : 0.0 for m in axes(Bᵗ₀)[1], n in axes(Bᵗ₀)[2], y in 1:2]
        b[t] = copy(Aᵗ)
    end
    return b |> TensorTrain
end