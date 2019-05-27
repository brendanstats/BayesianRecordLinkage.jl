struct ParameterChain{G <: Integer, T <: AbstractFloat}
    C::Array{G, 2}
    nlinks::Union{Array{G, 1}, Array{G, 2}}
    pM::Array{T, 2}
    pU::Array{T, 2}
    nsteps::Int
    linktrace::Bool
end

function counts2indicies(A::Array{G, 2}) where G <: Integer
    rows = Int[]
    cols = Int[]
    vals = Int[]
    for jj in 1:size(A, 2)
        for ii in 1:size(A, 1)
            if !iszero(A[ii, jj])
                push!(rows, ii)
                push!(cols, jj)
                push!(vals, A[ii, jj])
            end
        end
    end
    return [rows cols vals]
end

function counts2indicies(A::Array{G, 3}) where G <: Integer
    rows = Int[]
    cols = Int[]
    stages = Int[]
    vals = Int[]
    for kk in 1:size(A, 3)
        for jj in 1:size(A, 2)
            for ii in 1:size(A, 1)
                if !iszero(A[ii, jj, kk])
                    push!(rows, ii)
                    push!(cols, jj)
                    push!(stages, kk)
                    push!(vals, A[ii, jj, kk])
                end
            end
        end
    end
    return [rows cols stages vals]
end

function get_linkcounts(pchain::ParameterChain{G, T}) where {G <: Integer, T <: AbstractFloat}
    if pchain.linktrace
        ctDict = DefaultDict{Tuple{Int, Int}, Int}(zero(Int))
        for ii in 1:size(pchain.C, 1)
            ctDict[(pchain.C[ii, 1], pchain.C[ii, 2])] += pchain.C[ii, end] + one(Int) - pchain.C[ii, end - 1]
        end
        outC = Array{Int, 2}(undef, length(ctDict), 3)
        ii = 0
        for (ky, vl) in pairs(ctDict)
            ii += 1
            outC[ii, 1] = ky[1]
            outC[ii, 2] = ky[2]
            outC[ii, 3] = vl
        end
           return outC 
    else
        return pchain.C
    end
end

function get_linkstagecounts(pchain::ParameterChain{G, T}) where {G <: Integer, T <: AbstractFloat}
        if pchain.linktrace
        ctDict = DefaultDict{Tuple{Int, Int, Int}, Int}(zero(Int))
        for ii in 1:size(pchain.C, 1)
            ctDict[(pchain.C[ii, 1], pchain.C[ii, 2], pchain.C[ii, 3])] += pchain.C[ii, end] + one(Int) - pchain.C[ii, end - 1]
        end
        outC = Array{Int, 2}(undef, length(ctDict), 4)
        ii = 0
        for (ky, vl) in pairs(ctDict)
            ii += 1
            outC[ii, 1] = ky[1]
            outC[ii, 2] = ky[2]
            outC[ii, 3] = ky[3]
            outC[ii, 4] = vl
        end
           return outC 
    else
        return pchain.C
    end
end
