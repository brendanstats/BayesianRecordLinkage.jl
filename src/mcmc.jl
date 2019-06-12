"""
    dropoutside!(compsum::SparseComparisonSummary{G, Tv, Ti}, cc::ConnectedComponents) where {G <: Integer, Tv <: Integer, Ti <: Integer}

Drop indicies outside of connected components, removing all values from compsum.obsidx[ii, jj] where cc.rowLabels[ii] != cc.colLabels[jj].

This does not affect the values of `compsum.counts` or `compsum.npairs` meaning inferences
performed with the resulting object will be the same, with the constraint that all that all
record pairs outside of the connected components cannot be linked.

See also: [`dropoutside`](@ref), [`randomwalk1_log_move_weights_sparse`](@ref), [`PosthocBlocks`](@ref)
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

"""
    dropoutside(compsum::ComparisonSummary{G, T}, phb::PosthocBlocks{A}) where {G <: Integer, T <: Integer, A <: Integer}

Drop indicies outside of connected components, removing all values from compsum.obsidx[ii, jj] where cc.rowLabels[ii] != cc.colLabels[jj].

The returned value is therefore converted to a `SparseComparisonSummary` from a
`ComparisonSummary`. This does not affect the values of `compsum.counts` or `compsum.npairs`
meaning inferences performed with the resulting object will be the same, with the constraint
that all record pairs outside of the connected components cannot be linked.

See also: [`dropoutside!`](@ref), [`randomwalk1_log_move_weights_sparse`](@ref), [`PosthocBlocks`](@ref)
"""
function dropoutside(compsum::ComparisonSummary{G, T}, phb::PosthocBlocks{A}) where {G <: Integer, T <: Integer, A <: Integer}
    rows = Int[]
    cols = Int[]
    vals = T[]
    for kk in one(A):phb.nblock
        for jj in phb.block2cols[kk]
            for ii in phb.block2rows[kk]
                push!(rows, ii)
                push!(cols, jj)
                push!(vals, compsum.obsidx[ii, jj])
            end
        end
    end
    obsidx = sparse(rows, cols, vals)
    return SparseComparisonSummary(obsidx, compsum.obsvecs, compsum.obsvecct, compsum.counts, compsum.obsct, compsum.misct, compsum.nlevels, compsum.cmap, compsum.levelmap, compsum.cadj, compsum.nrow, compsum.ncol, compsum.npairs, compsum.ncomp)
end

"""
    mh_gibbs_count(nsteps::Integer, C0::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                   phb::PosthocBlocks, priorM::Array{T, 1}, priorU::Array{T, 1},
                   logpCRatio::Union{Function, Array{<:AbstractFloat, 1}}, transitionC!::Function,
                   loglikMissing::AbstractFloat = -Inf) where {T <: Real}

Restricted MCMC algorithm for drawing posterior samples of link distribition.

Returns a `ParameterChain` containing number of times each recordpairs was linked as well as
traces for the number of links, the M parameters, and the U parameters. A count of the number
of times each block structure is also returned as well as the final state of the link structure.
Each step involves an update to each block as defined in `phb` as well as a gibbs updates
for the M and U parameters by calling `gibbs_MU_draw`. For singleton blocks, those containing
 a single row and a single column a gibbs update to the link structure is automatically
performed using `singleton_gibbs!`,otherwise the specified update method is used.

# Arguments
* `nsteps::Integer`: Number of steps to run MCMC algorithm for, each step consistents up an
update to each block of `C` as defined in `phb` and a gibbs update to the matching parameters.
* `C0::LinkMatrix`: Parameter containing initial state of linkage structure.  Links should be contained within blocks of `phb`.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `phb::PosthocBlocks`: Blocking scheme restricting proposed links to those within blocks.
A version covering the entire space can be constructed with `PosthocBlocks(compsum)`
* `priorM::Array{<: Real, 1}`: Parameters of dirichlet prior on each comparison variable appended into a single vector.
* `priorU::Array{<: Real, 1}`: Parameters of dirichlet prior on each comparison variable appended into a single vector.
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios.
* `transitionC!::Function`: Function for updating link structure (e.g.randomwalk1_locally_balanced_sqrt_update!, randomwalk1_update!)
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also [`mh_gibbs_trace`](@ref), [`ParameterChain`](@ref), [`singleton_gibbs!`](@ref), [`gibbs_MU_draw`](@ref)
"""
function mh_gibbs_count(nsteps::Integer, C0::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
    phb::PosthocBlocks, priorM::Array{T, 1}, priorU::Array{T, 1},
    logpCRatio::Union{Function, Array{<:AbstractFloat, 1}}, transitionC!::Function,
    loglikMissing::AbstractFloat = -Inf) where {T <: Real}
    
    #MCMC Chains
    CArray = spzeros(Int, C0.nrow, C0.ncol)
    nlinkArray = Array{Int}(undef, nsteps)
    MArray = Array{Float64}(undef, length(priorM), nsteps)
    UArray = Array{Float64}(undef, length(priorU), nsteps)
    transC = zeros(Int, phb.nblock)
    
    ##Initial States
    countDeltas = get_obsidxcounts(compsum) #each column is an observation
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

"""
    mh_gibbs_count(nsteps::Integer, C0::LinkMatrix, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                   phb::PosthocBlocks, priorM::Array{T, 1}, priorU::Array{T, 1},
                   logpCRatio::Union{Function, Array{<:AbstractFloat, 1}}, transitionC!::Function,
                   loglikMissing::AbstractFloat = -Inf) where {T <: Real}

Restricted MCMC algorithm for drawing posterior samples of link distribition.

Returns a `ParameterChain` containing the trace of the entire link structure as well as
traces for the number of links, the M parameters, and the U parameters. A count of the number
of times each block structure is also returned as well as the final state of the link structure.
Each step involves an update to each block as defined in `phb` as well as a gibbs updates
for the M and U parameters by calling `gibbs_MU_draw`. For singleton blocks, those containing
 a single row and a single column a gibbs update to the link structure is automatically
performed using `singleton_gibbs!`,otherwise the specified update method is used.

# Arguments
* `nsteps::Integer`: Number of steps to run MCMC algorithm for, each step consistents up an
update to each block of `C` as defined in `phb` and a gibbs update to the matching parameters.
* `C0::LinkMatrix`: Parameter containing initial state of linkage structure.  Links should be contained within blocks of `phb`.
* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of record pair comparisons used in model fitting.
* `phb::PosthocBlocks`: Blocking scheme restricting proposed links to those within blocks.
A version covering the entire space can be constructed with `PosthocBlocks(compsum)`
* `priorM::Array{<: Real, 1}`: Parameters of dirichlet prior on each comparison variable appended into a single vector.
* `priorU::Array{<: Real, 1}`: Parameters of dirichlet prior on each comparison variable appended into a single vector.
* `logpCRatio::Union{Function, Array{T, 1}}`: Function or Array of logged prior ratios.
* `transitionC!::Function`: Function for updating link structure (e.g.randomwalk1_locally_balanced_sqrt_update!, randomwalk1_update!)
* `loglikMissing::AbstractFloat = -Inf` loglikelihood ratio to be applied for adding a missing value denomted by compsum.obsidx[row, col] == 0.

See also [`mh_gibbs_count`](@ref), [`ParameterChain`](@ref), [`singleton_gibbs!`](@ref), [`gibbs_MU_draw`](@ref)
"""
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
    countDeltas = get_obsidxcounts(compsum) #each column is an observation
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
