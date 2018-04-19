"""
    E_step(pM, pU, p, comparisonSummary) -> gM, gU

Expectation step in EM algorithm for estimating mixture model of matching and
non-matching record pairs.  
"""
function E_step{T <: AbstractFloat}(pM::Array{T, 1}, pU::Array{T, 1},
                                    p::AbstractFloat, compsum::Union{ComparisonSummary, SparseComparisonSummary})
    lpM = log.(pM)
    lpU = log.(pU)
    lpMatch = zeros(T, length(compsum.obsvecct))
    lpNon = zeros(T, length(compsum.obsvecct))
    for jj in 1:length(compsum.obsvecct), ii in 1:compsum.ncomp
        if compsum.obsvecs[ii, jj] != 0 #zeros are missing values and are integrated out
            lpMatch[jj] += lpM[compsum.cadj[ii] + compsum.obsvecs[ii, jj]]
            lpNon[jj] += lpU[compsum.cadj[ii] + compsum.obsvecs[ii, jj]]
        end
    end
    
    pMatch = exp.(log(p) .+ lpMatch)
    pNon = exp.(log(1.0 - p) .+ lpNon)
    tot = pMatch + pNon
    gM = pMatch ./ tot
    gU = pNon ./ tot
    return gM, gU
end

"""
    M_step(gM, gU, comparisonSummary, pseudoCountsM, pseudoCountsU) -> pM, pU, p

M step in EM algorithm.  Returns maximized parameter estimates including pseudo counts supplied for regularization.
"""
function M_step{T <: AbstractFloat, G <: Real}(gM::Array{T, 1}, gU::Array{T, 1},
                                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                               pseudoM::Array{G, 1},
                                               pseudoU::Array{G, 1})
    freqgM = compsum.obsvecct .* gM
    freqgU = compsum.obsvecct .* gU
    countsM = zeros(T, length(compsum.counts))
    countsU = zeros(T, length(compsum.counts))
    obsM = zeros(T, length(compsum.obsct))
    obsU = zeros(T, length(compsum.obsct))
    for jj in 1:length(compsum.obsvecct), ii in 1:compsum.ncomp
        if compsum.obsvecs[ii, jj] != 0
            countsM[compsum.cadj[ii] + compsum.obsvecs[ii, jj]] += freqgM[jj]
            obsM[ii] += freqgM[jj]

            countsU[compsum.cadj[ii] + compsum.obsvecs[ii, jj]] += freqgU[jj]
            obsU[ii] += freqgU[jj]
        end
    end

    for ii in 1:length(compsum.counts)
        obsM[compsum.cmap[ii]] += pseudoM[ii]
        obsU[compsum.cmap[ii]] += pseudoU[ii]
    end
    pM = (countsM + pseudoM) ./ obsM[compsum.cmap]
    pU = (countsU + pseudoU) ./ obsU[compsum.cmap]
    p = sum(freqgM) / compsum.npairs
    return pM, pU, p
end

"""
    estiamte_EM(pM0, pU0, p0, comparisonSummary, maxIterm pseduoCountsM, pseudCountsU, tol) -> pM, pU, p, iter, convergence

Resturns EM parameter estimates, the number of iterations and true/false value indicating if convergence was achieved.  
"""
function estimate_EM{T <: AbstractFloat}(pM0::Array{T, 1}, pU0::Array{T, 1},
                                         p0::AbstractFloat, compsum::Union{ComparisonSummary, SparseComparisonSummary};
                                         maxIter::Integer = 5000,
                                         pseudoM::Array{<:Real, 1} = mapreduce(x -> fill(1.0/x,x), append!, compsum.nlevels),
                                         pseudoU::Array{<:Real, 1} = mapreduce(x -> fill(1.0/x,x), append!, compsum.nlevels),
                                         tol::AbstractFloat = 1.0e-6)
    pM = copy(pM0)
    pU = copy(pU0)
    p = copy(p0)
    iter = 0
    while iter < maxIter
        iter += 1
        gM, gU = E_step(pM, pU, p, compsum)
        pMt, pUt, pt = M_step(gM, gU, compsum, pseudoM, pseudoU)
        maxM = maximum(abs, pM - pMt)
        maxU = maximum(abs, pU - pUt)
        maxp = abs(p - pt)
        if (maxM + maxU + maxp) < tol
            return pMt, pUt, pt, iter, true
        else
            pM = pMt
            pU = pUt
            p = pt
        end
    end
    println("Maximum Number of Iterations Reached")
    return pM, pU, p, iter, false
end
