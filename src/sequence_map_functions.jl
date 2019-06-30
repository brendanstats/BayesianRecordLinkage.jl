"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function next_penalty(pM::Array{T, 1},
                      pU::Array{T, 1},
                      compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      penalty0::AbstractFloat, mininc::AbstractFloat = 0.0) where T <: AbstractFloat
    wv = sort!(weights_vector(pM, pU, compsum))
    if iszero(mininc)
        error("mininc must be non-zero")
    elseif mininc > 0
        ii = findfirst(x -> (x - penalty0) > mininc, wv)
        if ii === nothing
            return wv[end], 0
        elseif ii < length(wv)
            return 0.95 * wv[ii] + 0.05 * wv[ii + 1], length(wv) - ii
        else
            return wv[ii], 1
        end
    else
        ii = findlast(x -> x <= (mininc + penalty0), wv)
        if ii === nothing
            return wv[1], length(w)
        elseif ii < length(wv)
            return 0.95 * wv[ii] + 0.05 * wv[ii + 1], ii
        else
            return wv[ii], 0
        end
    end
end

function next_penalty(row2col::Array{G, 1},
                      pM::Array{T, 1},
                      pU::Array{T, 1},
                      compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      penalty0::AbstractFloat, mininc::AbstractFloat = 0.0) where {G <: Integer, T <: AbstractFloat}
    matchobs = falses(length(compsum.obsvecct))
    for row in 1:length(row2col)
        if !iszero(row2col[row]) && row2col[row] <= compsum.ncol
            if !matchobs[compsum.obsidx[row, row2col[row]]]
                matchobs[compsum.obsidx[row, row2col[row]]] = true
            end
        end
    end

    if count(matchobs) == 0
        return penalty0, 0
   end
    
    wv = weights_vector(pM, pU, compsum)
    minmatch = minimum(wv[matchobs])
    sort!(wv)
    if iszero(mininc)
        error("mininc must be non-zero")
    elseif mininc > 0
        ii = findfirst(x -> (x - penalty0) > mininc && x >= minmatch, wv)
        if ii === nothing
            return wv[end], 0
        elseif ii < length(wv)
            return 0.95 * wv[ii] + 0.05 * wv[ii + 1], length(wv) - ii
        else
            return wv[ii], 1
        end
    else
        ii = findlast(x -> x <= (mininc + penalty0), wv)
        if ii === nothing
            return wv[1], length(w)
        elseif ii < length(wv)
            return 0.95 * wv[ii] + 0.05 * wv[ii + 1], ii
        else
            return wv[ii], 0
        end
    end
end

"""
    map_solver_iter(pM0, pU0, comparisonSummary, [priorM], [priorU], penaltyRng; maxIter) -> matchRows, matchColumns, pM, pU, iterations
"""
function map_solver_iter(pM0::Array{G, 1},
                         pU0::Array{G, 1},
                         compsum::ComparisonSummary{<:Integer, <:Integer},
                         priorM::Array{T, 1},
                         priorU::Array{T, 1},
                         penaltyRng::AbstractRange;
                         maxIter::Integer = 100,
                         verbose::Bool = false,
                         logfile::String = "log.txt",
                         logflag::Bool = false) where {G <: AbstractFloat, T <: Real}
    #Initialize variables
    npenalty = length(penaltyRng)
    outM = Array{G}(undef, npenalty, length(pM0))
    outU = Array{G}(undef, npenalty, length(pU0))
    outIter = Array{Int64}(undef, npenalty)
    
    #Solver for first value
    mrows, mcols, pM, pU, iter = map_solver(pM0, pU0, compsum, priorM, priorU, penaltyRng[1], maxIter = maxIter)
    outM[1, :] = pM
    outU[1, :] = pU
    outIter[1] = iter
    outMatches = Dict(1 => (mrows, mcols))
    
    for ii in 2:npenalty
        if verbose
            penalty = penaltyRng[ii]
            println("Step: $ii, Penalty $penalty")
        end
        if logflag
            penalty = penaltyRng[ii]
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "Step: $ii, Penalty $penalty\n\n")
            end
        end
        mrows, mcols, pM, pU, iter = map_solver(outM[ii - 1, :], outU[ii - 1, :], compsum, priorM, priorU, penaltyRng[ii], maxIter = maxIter)
        if length(mrows)  .== 0
            warn("No matches found with penalty of $(penaltyRng[ii])")
            return outMatches, outM, outU, outIter
        end
        outM[ii, :] = pM
        outU[ii, :] = pU
        outIter[ii] = iter
        outMatches[ii] = (mrows, mcols)
    end
    return outMatches, outM, outU, outIter
end

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function map_solver_iter_cluster(pM0::Array{G, 1},
                                 pU0::Array{G, 1},
                                 compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                 priorM::Array{T, 1},
                                 priorU::Array{T, 1},
                                 penaltyRng::AbstractRange;
                                 maxIter::Integer = 100,
                                 verbose::Bool = false,
                                 logfile::String = "log.txt",
                                 logflag::Bool = false) where {G <: AbstractFloat, T <: Real}
    #Initialize variables
    npenalty = length(penaltyRng)
    outM = Array{G}(undef, npenalty, length(pM0))
    outU = Array{G}(undef, npenalty, length(pU0))
    outIter = Array{Int64}(undef, npenalty)
    
    #Solver for first value
    mrows, mcols, pM, pU, iter = map_solver_cluster(pM0, pU0, compsum, priorM, priorU, penaltyRng[1], maxIter = maxIter, verbose = verbose)
    outM[1, :] = pM
    outU[1, :] = pU
    outIter[1] = iter
    outMatches = Dict(1 => (mrows, mcols))
    
    for ii in 2:npenalty
        if verbose
            penalty = penaltyRng[ii]
            println("Step: $ii, Penalty $penalty")
        end
        if logflag
            penalty = penaltyRng[ii]
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "Step: $ii, Penalty $penalty\n\n")
            end
        end
        mrows, mcols, pM, pU, iter = map_solver_cluster(outM[ii - 1, :], outU[ii - 1, :], compsum, priorM, priorU, penaltyRng[ii], maxIter = maxIter, verbose = verbose)
        if length(mrows)  .== 0
            warn("No matches found with penalty of $(penaltyRng[ii])")
            return outMatches, outM, outU, outIter
        end
        outM[ii, :] = pM
        outU[ii, :] = pU
        outIter[ii] = iter
        outMatches[ii] = (mrows, mcols)
    end
    return outMatches, outM, outU, outIter
end

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function map_solver_search(pM0::Array{G, 1}, pU0::Array{G, 1},
                           compsum::Union{ComparisonSummary, SparseComparisonSummary},
                           priorM::Array{T, 1}, priorU::Array{T, 1},
                           penalty0::Real = 0.0, mininc::Real = 0.0;
                           maxIter::Integer = 100,
                           verbose::Bool = false,
                           logfile::String = "log.txt",
                           logflag::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    outrows = Int[]
    outcols = Int[]
    outstart = Int[]
    outstop = Int[]
    
    #Solver for first value
    row2col, pM, pU, iter = map_solver(pM0, pU0, compsum, priorM, priorU, penalty0, maxIter = maxIter)
    penalty = copy(penalty0)
    nabove = count(penalized_weights_vector(pM, pU, compsum, penalty0) .> 0.0)
    outLinks = [count(.!iszero.(row2col))]
    outM = copy(pM)
    outU = copy(pU)
    outIter = [iter]
    penalties = [penalty0]

    prevrow2col = copy(row2col)
    startrow2col = ones(Int, length(row2col)) #zeros(Int, length(currrow2col))
    
    ii = 1
    
    while nabove > 1
        if verbose
            println("penalty: $penalty, matches: $(length(mrows)), nabove: $nabove")
        end
        if logflag
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "penalty: $penalty, matches: $(length(mrows)), nabove: $nabove\n\n")
            end
        end
        ii += 1
        penalty, nabove = next_penalty(row2col, pM, pU, compsum, penalty, mininc)

        #Delete matches that would be excluded by the increased penalty
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        for row in 1:length(row2col)
            if !iszero(row2col[row]) && row2col[row] < compsum.ncol
                if iszero(w[compsum.obsidx[row, row2col[row]]])
                    row2col[row] = 0
                end
            end
        end
        pM, pU = max_MU(row2col, compsum, pseudoM, pseudoU)

        row2col, pM, pU, iter = map_solver(pM, pU, compsum, priorM, priorU, penalty, maxIter = maxIter)

        for row in 1:length(row2col)
            if prevrow2col[row] != row2col[row]
                #record if deletion or move (not additions)
                if !iszero(prevrow2col[row])
                    push!(outrows, row)
                    push!(outcols, prevrow2col[row])
                    push!(outstart, startrow2col[row])
                    push!(outstop, ii - 1)                    
                end
                
                prevrow2col[row] = row2col[row]
                startrow2col[row] = ii
            end
        end
        
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        push!(outLinks, count(.!iszero.(row2col)))
        outIter = push!(outIter, iter)
        #outMatches[ii] = copy(row2col)
        push!(penalties, penalty)

        nabove = count(penalized_weights_vector(pM, pU, compsum, penalty) .> 0.0)
    end

    for row in 1:length(row2col)
        if !iszero(row2col[row])
            push!(outrows, row)
            push!(outcols, prevrow2col[row])
            push!(outstart, startrow2col[row])
            push!(outstop, ii)                    
        end
    end
    
    #return outMatches, permutedims(outM, [2, 1]), permutedims(outU, [2, 1]), penalties, outIter
    return ParameterChain([outrows outcols outstart outstop], outLinks, permutedims(outM, [2, 1]), permutedims(outU, [2, 1]), length(outLinks), true), penalties, outIter
end

function penalized_likelihood_search_hungarian(pM0::Array{G, 1}, pU0::Array{G, 1},
                                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                               priorM::Array{T, 1}, priorU::Array{T, 1},
                                               penalty0::Real = 0.0, mininc::Real = 0.0;
                                               tol::AbstractFloat = 0.0, maxIter::Integer = 100,
                                               cluster::Bool = true, verbose::Bool = false,
                                               logfile::String = "log.txt",
                                               logflag::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    outrows, outcols, outstart, outstop = Int[], Int[], Int[], Int[]
        
    #Solver for first value
    astate, pM, pU, iter = penalized_likelihood_hungarian(pM0, pU0, compsum, priorM, priorU, penalty0, tol = tol, maxIter = maxIter, cluster = cluster)
    penalty = copy(penalty0)
    nabove = count(penalized_weights_vector(pM, pU, compsum, penalty0) .> 0.0)
    outLinks = [astate.nassigned]
    outM = reshape(pM, (length(pM), 1))
    outU = reshape(pU, (length(pM), 1))
    outIter = [iter]
    penalties = [penalty0]

    prevrow2col = copy(astate.r2c)
    startrow2col = ones(Int, astate.nrow) #zeros(Int, length(currrow2col))
    
    ii = 1
    
    while nabove > 1
        if verbose
            println("penalty: $penalty, matches: $(astate.nassigned), nabove: $nabove")
        end
        if logflag
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "penalty: $penalty, matches: $(astate.nassigned), nabove: $nabove\n\n")
            end
        end
        ii += 1
        penalty, nabove = next_penalty(astate.r2c, pM, pU, compsum, penalty, mininc)

        #Delete matches that would be excluded by the increased penalty
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        for row in 1:astate.nrow
            if !iszero(astate.r2c[row]) #should be handled... [row] < compsum.ncol
                if iszero(w[compsum.obsidx[row, astate.r2c[row]]])
                    astate.c2r[astate.r2c[row]] = 0
                    astate.r2c[row] = 0
                    astate.nassigned -= 1
                end
            end
        end
        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU)

        astate, pM, pU, iter = penalized_likelihood_hungarian(pM, pU, compsum, priorM, priorU, penalty, tol = tol, maxIter = maxIter, cluster = cluster)

        for row in 1:astate.nrow
            if prevrow2col[row] != astate.r2c[row]
                
                #record if deletion or move (not additions)
                if !iszero(prevrow2col[row])
                    push!(outrows, row)
                    push!(outcols, prevrow2col[row])
                    push!(outstart, startrow2col[row])
                    push!(outstop, ii - 1)                    
                end
                
                prevrow2col[row] = astate.r2c[row]
                startrow2col[row] = ii
            end
        end
        
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        push!(outLinks, astate.nassigned)
        outIter = push!(outIter, iter)
        push!(penalties, penalty)

        nabove = count(penalized_weights_vector(pM, pU, compsum, penalty) .> 0.0)
    end

    for row in 1:astate.nrow
        if !iszero(astate.r2c[row])
            push!(outrows, row)
            push!(outcols, prevrow2col[row])
            push!(outstart, startrow2col[row])
            push!(outstop, ii)                    
        end
    end
    
    return ParameterChain([outrows outcols outstart outstop], outLinks, permutedims(outM), permutedims(outU), length(outLinks), true), penalties, outIter
end

function penalized_likelihood_search_auction(pM0::Array{G, 1}, pU0::Array{G, 1},
                                             compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                             priorM::Array{T, 1}, priorU::Array{T, 1},
                                             penalty0::Real = 0.0, mininc::Real = 0.0;
                                             epsiscale::T = T(0.2), minmargin::T = zero(T), digt::Integer = 5,
                                             tol::AbstractFloat = 0.0, maxIter::Integer = 100,
                                             cluster::Bool = true, update::Bool = true, verbose::Bool = false,
                                             logfile::String = "log.txt",
                                             logflag::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))

    outrows, outcols, outstart, outstop = Int[], Int[], Int[], Int[]
        
    #Solver for first value
    astate, pM, pU, iter = penalized_likelihood_auction(pM0, pU0, compsum, priorM, priorU, penalty0, epsiscale = epsiscale, minmargin = minmargin, digt = digt, tol = tol, maxIter = maxIter, cluster = cluster, update = update)
    penalty = copy(penalty0)
    nabove = count(penalized_weights_vector(pM, pU, compsum, penalty0) .> 0.0)
    outLinks = [astate.nassigned]
    outM = reshape(pM, (length(pM), 1))
    outU = reshape(pU, (length(pM), 1))
    outIter = [iter]
    penalties = [penalty0]

    prevrow2col = copy(astate.r2c)
    startrow2col = ones(Int, astate.nrow) #zeros(Int, length(currrow2col))
    
    ii = 1
    
    while nabove > 1
        if verbose
            println("penalty: $penalty, matches: $(astate.nassigned), nabove: $nabove")
        end
        if logflag
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "penalty: $penalty, matches: $(astate.nassigned), nabove: $nabove\n\n")
            end
        end
        ii += 1
        penalty, nabove = next_penalty(astate.r2c, pM, pU, compsum, penalty, mininc)

        #Delete matches that would be excluded by the increased penalty
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        for row in 1:astate.nrow
            if !iszero(astate.r2c[row]) #should be handled... [row] < compsum.ncol
                if iszero(w[compsum.obsidx[row, astate.r2c[row]]])
                    astate.c2r[astate.r2c[row]] = 0
                    astate.r2c[row] = 0
                    astate.nassigned -= 1
                end
            end
        end
        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU)

        astate, pM, pU, iter = penalized_likelihood_auction(pM, pU, compsum, priorM, priorU, penalty, epsiscale = epsiscale, minmargin = minmargin, digt = digt, tol = tol, maxIter = maxIter, cluster = cluster, update = update)

        for row in 1:astate.nrow
            if prevrow2col[row] != astate.r2c[row]
                
                #record if deletion or move (not additions)
                if !iszero(prevrow2col[row])
                    push!(outrows, row)
                    push!(outcols, prevrow2col[row])
                    push!(outstart, startrow2col[row])
                    push!(outstop, ii - 1)                    
                end
                
                prevrow2col[row] = astate.r2c[row]
                startrow2col[row] = ii
            end
        end
        
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        push!(outLinks, astate.nassigned)
        outIter = push!(outIter, iter)
        push!(penalties, penalty)

        nabove = count(penalized_weights_vector(pM, pU, compsum, penalty) .> 0.0)
    end

    for row in 1:astate.nrow
        if !iszero(astate.r2c[row])
            push!(outrows, row)
            push!(outcols, prevrow2col[row])
            push!(outstart, startrow2col[row])
            push!(outstop, ii)                    
        end
    end
    
    return ParameterChain([outrows outcols outstart outstop], outLinks, permutedims(outM), permutedims(outU), length(outLinks), true), penalties, outIter
end

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function map_solver_search_cluster(pM0::Array{G, 1},
                                   pU0::Array{G, 1},
                                   compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                   priorM::Array{T, 1},
                                   priorU::Array{T, 1},
                                   penalty0::Real = 0.0,
                                   mininc::Real = 0.0;
                                   maxIter::Integer = 100,
                                   verbose::Bool = false,
                                   logfile::String = "log.txt",
                                   logflag::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    
    #Solver for first value
    mrows, mcols, pM, pU, iter = map_solver_cluster(pM0, pU0, compsum, priorM, priorU, penalty0, maxIter = maxIter, verbose = verbose)
    outM = copy(pM)
    outU = copy(pU)
    outIter = [iter]
    penalties = [penalty0]
    outMatches = Dict(1 => (mrows, mcols))
    penalty, nabove = next_penalty(mrows, mcols, pM, pU, compsum, penalty0, mininc)
    ii = 1
    while nabove > 1
        if verbose
            println("penalty: $penalty, matches: $(length(mrows)), nabove: $nabove")
        end
        if logflag
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "penalty: $penalty, matches: $(length(mrows)), nabove: $nabove\n\n")
            end
        end
        ii += 1
        mrows, mcols, pM, pU, iter = map_solver_cluster(pM, pU, compsum, priorM, priorU, penalty, maxIter = maxIter, verbose = verbose)
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        outIter = push!(outIter, iter)
        outMatches[ii] = (mrows, mcols)
        push!(penalties, penalty)
        penalty, nabove = next_penalty(mrows, mcols, pM, pU, compsum, penalty, mininc)

        #Delete matches that would be excluded by the increased penalty
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        keep = [w[compsum.obsidx[row, col]] > 0.0 for (row, col) in zip(mrows, mcols)]
        mrows = mrows[keep]
        mcols = mcols[keep]
        pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)

    end
    return outMatches, permutedims(outM, [2, 1]), permutedims(outU, [2, 1]), penalties, outIter
end

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function map_solver_search_auction(pM0::Array{G, 1},
                                   pU0::Array{G, 1},
                                   compsum::ComparisonSummary{<:Integer, <:Integer},
                                   priorM::Array{T, 1},
                                   priorU::Array{T, 1},
                                   penalty0::Real = 0.0,
                                   mininc::Real = 0.0,
                                   εscale::T = 0.2;
                                   maxIter::Integer = 100,
                                   verbose::Bool = false,
                                   logfile::String = "log.txt",
                                   logflag::Bool = false) where {G <: AbstractFloat, T <: Real}
    
    ##Modes are found using pseudo counts of αᵢ - 1
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    penalty = copy(penalty0)
    
    ##Allocate variables
    outM = Array{T}(undef, length(pM0), 0)
    outU = Array{T}(undef, length(pU0), 0)
    outIter = Array{Int}(undef, 0)
    penalties = Array{typeof(penalty)}(undef, 0)
    outMatches = Dict{Int, Tuple{Array{Int, 1}, Array{Int, 1}}}()

    #Solver for first value
    mrows, mcols, rowCosts, colCosts, prevw, prevε, λ = max_C_auction(pM0, pU0, compsum, penalty, εscale)
    pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)

    #track match counts for convergence
    prevcounts = counts_matches(mrows, mcols, compsum)[1]
    nabove = count(weights_vector(pM, pU, compsum) .> penalty)
    
    ii = 0
    while nabove > 1
        ii += 1
        iter = 0
        while iter < maxIter
            iter += 1

            #solve new problem
            mrows, mcols, rowCosts, colCosts, prevw, prevε, λ = max_C_auction(pM, pU, compsum, rowCosts, colCosts, prevw, prevε, λ, penalty, εscale)

            #check convergence
            matchcounts = counts_matches(mrows, mcols, compsum)[1]
            if prevcounts == matchcounts
                break
            end
            prevcounts = matchcounts

            #update parameters
            pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)

        end

        ##Add values to outputs
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        outIter = push!(outIter, iter)
        outMatches[ii] = (mrows, mcols)
        push!(penalties, penalty)
        
        if verbose
            println("penalty: $penalty, matches: $(length(mrows)), nabove: $nabove")
        end
        if logflag
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "penalty: $penalty, matches: $(length(mrows)), nabove: $nabove\n\n")
            end
        end
        
        ##Find next penalty
        penalty, nabove = next_penalty(mrows, mcols, pM, pU, compsum, penalty, mininc)

        #Delete matches that would be excluded by the increased penalty
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        keep = [w[compsum.obsidx[row, col]] > 0.0 for (row, col) in zip(mrows, mcols)]
        mrows = mrows[keep]
        mcols = mcols[keep]
        pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)
        prevcounts = counts_matches(mrows, mcols, compsum)[1]
    end
    return outMatches, permutedims(outM, [2, 1]), permutedims(outU, [2, 1]), penalties, outIter
end

"""
    f(x::Type)

### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
function map_solver_search_auction_cluster(pM0::Array{G, 1},
                                           pU0::Array{G, 1},
                                           compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                           priorM::Array{T, 1},
                                           priorU::Array{T, 1},
                                           penalty0::Real = 0.0,
                                           mininc::Real = 0.0,
                                           εscale::T = 0.2;
                                           maxIter::Integer = 100,
                                           verbose::Bool = false,
                                           logfile::String = "log.txt",
                                           logflag::Bool = false) where {G <: AbstractFloat, T <: Real}
    ##Modes are found using pseudo counts of αᵢ - 1
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    penalty = copy(penalty0)
    
    ##Allocate variables
    outM = Array{T}(undef, length(pM0), 0)
    outU = Array{T}(undef, length(pU0), 0)
    outIter = Array{Int}(undef, 0)
    penalties = Array{typeof(penalty)}(undef, 0)
    outMatches = Dict{Int, Tuple{Array{Int, 1}, Array{Int, 1}}}()

    tractable = false
    while !tractable
        w = penalized_weights_vector(pM0, pU0, compsum, penalty)
        weightMat = dropzeros(sparse(penalized_weights_matrix(w, compsum)))
        rowLabels, colLabels, maxLabel = bipartite_cluster(weightMat)
        concomp = ConnectedComponents(rowLabels, colLabels, maxLabel)
        if maximum(concomp.rowcounts[2:end] .* concomp.colcounts[2:end]) < 1000000
            tractable = true
        else
            penalty += max(mininc, 0.01)
        end
    end
    
    #Solver for first value
    mrows, mcols, r2c, c2r, rowCosts, colCosts, prevw, prevmargin = max_C_auction_cluster(pM0, pU0, compsum, penalty, εscale, verbose = verbose)
    pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)

    #track match counts for convergence
    prevcounts = counts_matches(mrows, mcols, compsum)[1]
    nabove = count(weights_vector(pM, pU, compsum) .> penalty)

    if logflag
        open(logfile, "a") do f
            write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
            write(f, "penalty: $penalty\n")
        end
    end
    
    ii = 0
    aboveLower = true
    while nabove > 1 && aboveLower
        ii += 1
        iter = 0
        while iter < maxIter
            iter += 1

            if logflag
                open(logfile, "a") do f
                    write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "; Iteration: $iter, matches: $(length(mrows))\n")
                end
            end

            tractable = false
            while !tractable
                w = penalized_weights_vector(pM, pU, compsum, penalty)
                weightMat = dropzeros(sparse(penalized_weights_matrix(w, compsum)))
                rowLabels, colLabels, maxLabel = bipartite_cluster(weightMat)
                concomp = ConnectedComponents(rowLabels, colLabels, maxLabel)
                if maximum(concomp.rowcounts[2:end] .* concomp.colcounts[2:end]) < 1000000
                    tractable = true
                else
                    penalty += max(mininc, 0.01)
                    iter = 1
                    if mininc < 0.0
                        aboveLower = false
                    end
                end
            end
            
            #solve new problem
            mrows, mcols, r2c, c2r, rowCosts, colCosts, prevw, prevmargin = max_C_auction_cluster(pM, pU, compsum, r2c, c2r, rowCosts, colCosts, prevw, prevmargin, penalty, εscale, verbose = verbose)

            #check convergence
            matchcounts = counts_matches(mrows, mcols, compsum)[1]
            if prevcounts == matchcounts
                break
            end
            prevcounts = matchcounts

            #update parameters
            pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)

        end

        ##Add values to outputs
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        outIter = push!(outIter, iter)
        outMatches[ii] = (mrows, mcols)
        push!(penalties, penalty)
        if verbose
            println("penalty: $penalty, matches: $(length(mrows)), nabove: $nabove")
        end
        if logflag
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "penalty: $penalty, matches: $(length(mrows)), nabove: $nabove\n\n")
            end
        end
        ##Find next penalty
        penalty, nabove = next_penalty(mrows, mcols, pM, pU, compsum, penalty, mininc)

        #Delete matches that would be excluded by the increased penalty
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        keep = [w[compsum.obsidx[row, col]] > 0.0 for (row, col) in zip(mrows, mcols)]
        mrows = mrows[keep]
        mcols = mcols[keep]
        pM, pU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)
        prevcounts = counts_matches(mrows, mcols, compsum)[1]
    end
    return outMatches, permutedims(outM, [2, 1]), permutedims(outU, [2, 1]), penalties, outIter
end
