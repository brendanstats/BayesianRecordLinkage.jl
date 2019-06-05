"""
Drop indicies outside of connected components, removing all values from compsum.obsidx[ii, jj] where cc.rowLabels[ii] != cc.colLabels[jj]
"""
function dropoutside!(compsum::SparseComparisonSummary{G, Tv, Ti}, cc::ConnectedComponents) where {G <: Integer, Tv <: Integer, Ti <: Integer}
    rows = rowvals(compsum.obsidx)
    vals = nonzeros(compsum.obsidx)
    for jj = 1:compsum.ncol
        for ii in nzrange(compsum.obsidx, jj)
            row = rows[ii]
            if iszero(cc.rowLabels[row]) || (cc.rowLabels[row] != cc.colLabels[jj])
                vals[ii] = zero(Ti)
            end
        end
    end
    dropzeros!(compsum.obsidx)
    return compsum
end

function mh_gibbs_count(
    nsteps::Integer,
    C0::LinkMatrix,
    compsum::Union{ComparisonSummary, SparseComparisonSummary},
    phb::PosthocBlocks{G},
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpCRatio::Union{Function, Array{<:AbstractFloat, 1}},
    transitionC!::Function,
    loglikMissing::AbstractFloat = -Inf) where {G <: Integer, T <: Real}
    
    #MCMC Chains
    CArray = spzeros(Int, C0.nrow, C0.ncol)
    nlinkArray = Array{Int}(undef, nsteps)
    MArray = Array{Float64}(undef, length(priorM), nsteps)
    UArray = Array{Float64}(undef, length(priorU), nsteps)
    transC = zeros(Int64, phb.nblock)
    
    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU, loglikRatios = gibbs_MU_draw(matchcounts, compsum, countDeltas, priorM, priorU)
    
    #Outer iteration (recorded)
    for ii in 1:nsteps

        #Loop over blocks
        for kk in 1:phb.nblock
            move = false
            countdelta = zeros(eltype(matchcounts), length(matchcounts))
            if (phb.blocknrows[kk] == 1) && (phb.blockncols[kk] == 1)
                C, countdelta, move = singleton_gibbs!(phb.block2rows[kk][1], phb.block2cols[kk][1], C, compsum, loglikRatios, countDeltas, logpCRatio, loglikMissing)
            else
                C, countdelta, move = transitionC!(phb.block2rows[kk], phb.block2cols[kk], C, compsum, loglikRatios, countDeltas, logpCRatio, loglikMissing)
            end
            if move
                transC[kk] += 1
                matchcounts += countdelta
            end
        end #end block loop

        ##Perform Gibbs update if performed with outer iterations
        pM, pU, loglikRatios = gibbs_MU_draw(matchcounts, compsum, countDeltas, priorM, priorU)

        #Add states to chain
        for row in 1:C.nrow
            if !iszero(C.row2col[row])
                CArray[row, C.row2col[row]] += 1
            end
        end
        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end
    ParameterChain(counts2indicies(CArray), nlinkArray, permutedims(MArray, [2, 1]), permutedims(UArray, [2, 1]), nsteps, false), transC, C
end

function mh_gibbs_trace(
    nsteps::Integer,
    C0::LinkMatrix,
    compsum::Union{ComparisonSummary, SparseComparisonSummary},
    phb::PosthocBlocks{G},
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpCRatio::Union{Function, Array{<:AbstractFloat, 1}},
    transitionC!::Function,
    loglikMissing::AbstractFloat = -Inf) where {G <: Integer, T <: Real}
    
    #MCMC Chains
    outrows = Int[]
    outcols = Int[]
    outstart = Int[]
    outstop = Int[]
    
    nlinkArray = Array{Int64}(undef, nsteps)
    MArray = Array{Float64}(undef, length(priorM), nsteps)
    UArray = Array{Float64}(undef, length(priorU), nsteps)
    transC = zeros(Int64, phb.nblock)
    
    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU, loglikRatios = gibbs_MU_draw(matchcounts, compsum, countDeltas, priorM, priorU)

    currrow2col = copy(C.row2col)
    startrow2col = ones(Int, length(currrow2col)) #zeros(Int, length(currrow2col))
    
    #Outer iteration (recorded)
    for ii in 1:nsteps

        #Loop over blocks
        for kk in 1:phb.nblock
            move = false
            countdelta = zeros(eltype(matchcounts), length(matchcounts))
            if (phb.blocknrows[kk] == 1) && (phb.blockncols[kk] == 1)
                C, countdelta, move = singleton_gibbs!(phb.block2rows[kk][1], phb.block2cols[kk][1], C, compsum, loglikRatios, countDeltas, logpCRatio, loglikMissing)
            else
                C, countdelta, move = transitionC!(phb.block2rows[kk], phb.block2cols[kk], C, compsum, loglikRatios, countDeltas, logpCRatio, loglikMissing)
            end
            if move
                transC[kk] += 1
                matchcounts += countdelta
            end
        end #end block loop
        
        ##Perform Gibbs update if performed with outer iterations
        pM, pU, loglikRatios = gibbs_MU_draw(matchcounts, compsum, countDeltas, priorM, priorU)
        
        #Add states to chain
        for row in 1:C.nrow
            if currrow2col[row] != C.row2col[row]
                #record if deletion or move (not additions)
                if !iszero(currrow2col[row])
                    push!(outrows, row)
                    push!(outcols, currrow2col[row])
                    push!(outstart, startrow2col[row])
                    push!(outstop, ii - 1)                    
                end
                
                currrow2col[row] = C.row2col[row]
                startrow2col[row] = ii
            end
        end
        
        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end

    #add current links to chain
    for row in 1:C.nrow
        if !iszero(C.row2col[row])
            push!(outrows, row)
            push!(outcols, currrow2col[row])
            push!(outstart, startrow2col[row])
            push!(outstop, nsteps)                    
        end
    end
    
    ParameterChain([outrows outcols outstart outstop][outstart .<= outstop, :], nlinkArray, permutedims(MArray, [2, 1]), permutedims(UArray, [2, 1]), nsteps, true), transC, C
end
