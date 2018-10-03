function next_penalty(pM::Array{T, 1},
                                          pU::Array{T, 1},
                                          compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                          penalty0::AbstractFloat, mininc::AbstractFloat = 0.0) where T <: AbstractFloat
    wv = sort!(weights_vector(pM, pU, compsum))
    ii = findfirst(x -> (x - penalty0) > mininc, wv)
    if ii == 0
        return wv[end], 0
    elseif ii < length(wv)
        return 0.95 * wv[ii] + 0.05 * wv[ii + 1], length(wv) - ii
    else
        return wv[ii], 1
    end
end

function next_penalty(mrows::Array{G, 1},
                                                        mcols::Array{G, 1},
                                                        pM::Array{T, 1},
                                                        pU::Array{T, 1},
                                                        compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                                        penalty0::AbstractFloat, mininc::AbstractFloat = 0.0) where {G <: Integer, T <: AbstractFloat}
    matchobs = falses(length(compsum.obsvecct))
    for (ii, jj) in zip(mrows, mcols)
        if !matchobs[compsum.obsidx[ii, jj]]
            matchobs[compsum.obsidx[ii, jj]] = true
        end
    end

    if count(matchobs) == 0
        return penalty0, 0
   end
    
    wv = sort!(weights_vector(pM, pU, compsum)[matchobs])
    ii = findfirst(x -> (x - penalty0) > mininc, wv)
    if ii == 0
        return wv[end], 0
    elseif ii < length(wv)
        return 0.95 * wv[ii] + 0.05 * wv[ii + 1], length(wv) - ii
    else
        return wv[ii], 1
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
    outM = Array{G}(npenalty, length(pM0))
    outU = Array{G}(npenalty, length(pU0))
    outIter = Array{Int64}(npenalty)
    
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
    outM = Array{G}(npenalty, length(pM0))
    outU = Array{G}(npenalty, length(pU0))
    outIter = Array{Int64}(npenalty)
    
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


function map_solver_iter_initialize(pM0::Array{G, 1},
                                                                   pU0::Array{G, 1},
                                                                   compsum::ComparisonSummary{<:Integer, <:Integer},
                                                                   priorM::Array{T, 1},
                                                                   priorU::Array{T, 1},
                                                                   penaltyRng::AbstractRange;
                                                                   maxIter::Integer = 100,
                                                                   verbose::Bool = false,
                                                                   logfile::String = "log.txt",
                                                                   logflag::Bool = false) where {G <: AbstractFloat, T <: Real}
    ##Modes are found using pseudo counts of αᵢ - 1
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    pM = copy(pM0)
    pU = copy(pU0)
    
    #Initialize variables
    npenalty = length(penaltyRng)
    outM = Array{G}(npenalty, length(pM0))
    outU = Array{G}(npenalty, length(pU0))
    outIter = Array{Int64}(npenalty)
    outMatches = Dict{Int64,Tuple{Array{Int64,1},Array{Int64,1}}}()
    #mrows, mcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_offsets(pM, pU, compsum, penaltyRng[1])
    currmrows, currmcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_offsets(pM, pU, compsum, penaltyRng[1])
    for (ii, penalty) in enumerate(IndexLinear(), penaltyRng)
        if verbose
            println("Step: $ii, Penalty $penalty")
        end
        if logflag
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "Step: $ii, Penalty $penalty\n\n")
            end
        end
        iter = 0
        while iter < maxIter
            iter += 1
            pM, pU = max_MU(currmrows, currmcols, compsum, pseudoM, pseudoU)
            #npM, npU = max_MU(mrows, mcols, compsum, pseudoM, pseudoU)
            #if all(npM .== pM) && all(npU .== pU)
            #    break
            #else
            #    pM = npM
            #    pU = npU
            #end
            newmrows, newmcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_initialized!(pM, pU, compsum, penalty, rows2cols, rowOffsets, colOffsets, maxcost)
            #mrows, mcols, rows2cols, rowOffsets, colOffsets, maxcost = max_C_initialized!(pM, pU, compsum, penalty, rows2cols, rowOffsets, colOffsets, maxcost)
            if length(newmrows) == length(currmrows)
                if all(newmrows .== currmrows) && all(newmcols .== currmcols)
                    break
                end
            end
            currmrows = newmrows
            currmcols = newmcols
        end
        if iter >= maxIter
            warn("Maximum number of iterations reached")
            if logflag
                open(logfile, "a") do f
                    write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                    write(f, "Maximum number of iterations reached\n\n")
                end
            end
        end
        outM[ii, :] = pM
        outU[ii, :] = pU
        outIter[ii] = iter
        outMatches[ii] = (currmrows, currmcols)
        #outMatches[ii] = (mrows, mcols)
    end
    return outMatches, outM, outU, outIter
end

function map_solver_search(pM0::Array{G, 1},
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
    mrows, mcols, pM, pU, iter = map_solver(pM0, pU0, compsum, priorM, priorU, penalty0, maxIter = maxIter)
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
        mrows, mcols, pM, pU, iter = map_solver(pM, pU, compsum, priorM, priorU, penalty, maxIter = maxIter)
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
    return outMatches, outM', outU', penalties, outIter
end

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
    return outMatches, outM', outU', penalties, outIter
end

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
    outM = Array{T}(length(pM0), 0)
    outU = Array{T}(length(pU0), 0)
    outIter = Array{Int}(0)
    penalties = Array{typeof(penalty)}(0)
    outMatches = Dict{Int,Tuple{Array{Int,1},Array{Int,1}}}()

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
    return outMatches, outM', outU', penalties, outIter
end

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
    outM = Array{T}(length(pM0), 0)
    outU = Array{T}(length(pU0), 0)
    outIter = Array{Int}(0)
    penalties = Array{typeof(penalty)}(0)
    outMatches = Dict{Int,Tuple{Array{Int,1},Array{Int,1}}}()

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
    while nabove > 1
        ii += 1
        iter = 0
        while iter < maxIter
            iter += 1

            if logflag
                open(logfile, "a") do f
                    write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "; Iteration: $iter, matches: $(length(mrows))\n")
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
    return outMatches, outM', outU', penalties, outIter
end
