"""
    incr_penalty(penalty::AbstractFloat, weightvec::Array{T, 1}, minnext::AbstractFloat = penalty; frac::AbstractFloat = 0.5, sorted::Bool = false)
    incr_penalty(pM::Array{T, 1}, pU::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      penalty::AbstractFloat, mininc::AbstractFloat = 0.0, frac::AbstractFloat = 0.5) where T <: AbstractFloat
    incr_penalty(row2col::Array{G, 1}, pM::Array{T, 1}, pU::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      penalty::AbstractFloat, mininc::AbstractFloat = 0.0, frac::AbstractFloat = 0.5) where {G <: Integer, T <: AbstractFloat}

Function for incrementing the penalty, internal to `penalized_likelihood_search_*`.  Version accepting a `weightvec` parameter is wrapped within the
other two and will sort `weightvec` (using sort!).  If row2col is not supplied the penalty will be incremented tobby at least `mininc`.  The two
weight values it falls between w[below] and w[above] will be averaged by w[below] * (1 -frac) + w[above] * frac and this will be the new penalty
value if it is greater than minnext (or penalty + mininc).
"""
function incr_penalty(penalty::AbstractFloat, weightvec::Array{T, 1}, minnext::AbstractFloat = penalty; frac::AbstractFloat = 0.5, sorted::Bool = false) where T <: AbstractFloat

    if minnext < penalty
        @error "minnext must be at least as large as penalty"
    end
    
    if length(weightvec) == 0
        penalty = minnext
        return penalty, 0
    end

    if length(weightvec) == 1
        penalty = minnext
        if weightvec[1] > minnext
            return penalty, 1
        else
            return penalty, 0
        end
    end

    if !sorted
        sort!(weightvec)
    end

    idx = findfirst(x -> x > minnext, weightvec)
    if idx == nothing
        penalty = minnext
        return penalty, 0
    elseif idx == 1
        penalty = minnext
        return penalty, length(weightvec)
    else
        penalty = max((1.0 - frac) * weightvec[idx - 1] + (frac) * weightvec[idx], minnext)
        return penalty, length(weightvec) - idx + 1
    end
end


function incr_penalty(pM::Array{T, 1}, pU::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary}, penalty::AbstractFloat, mininc::AbstractFloat = 0.0, frac::AbstractFloat = 0.5) where T <: AbstractFloat

    if mininc < 0.0
        @error "mininc must be non-negative"
    end

    wv = filter!(x -> x > penalty, weights_vector(pM, pU, compsum))
    wv = sort!(wv)
    
    minnext = max(penalty + mininc, wv[1])
    return incr_penalty(penalty, wv, minnext, frac = frac, sorted = true)    
end

function incr_penalty(row2col::Array{G, 1}, pM::Array{T, 1}, pU::Array{T, 1}, compsum::Union{ComparisonSummary, SparseComparisonSummary},
                      penalty::AbstractFloat, mininc::AbstractFloat = 0.0, frac::AbstractFloat = 0.5) where {G <: Integer, T <: AbstractFloat}

    if mininc < 0.0
        @error "mininc must be non-negative"
    end

    matchobs = matched_comparisons(row2col, compsum)
    nuniqueobs = count(matchobs)
    if nuniqueobs == 0
        return penalty + mininc, 0
    end
        
    wv = weights_vector(pM, pU, compsum)
    
    minmatch, maxmatch = extrema(wv[matchobs])
    if maxmatch <= (penalty + mininc)
        return penalty + mininc, 0
    end
    
    wv = filter!(x -> x > penalty, wv)
    wv = sort!(wv)
    
    minnext = max(penalty + mininc, minmatch)
    return incr_penalty(penalty, wv, minnext, frac = frac, sorted = true)    
end

function penalized_likelihood_search_hungarian(pM0::Array{G, 1}, pU0::Array{G, 1},
                                               compsum::Union{ComparisonSummary, SparseComparisonSummary},
                                               priorM::Array{T, 1}, priorU::Array{T, 1},
                                               penalty0::Real = 0.0, mininc::Real = 0.0;
                                               tol::AbstractFloat = 0.0, maxIter::Integer = 100,
                                               cluster::Bool = true, verbose::Bool = false,
                                               logfile::String = "log.txt",
                                               logflag::Bool = false) where {G <: AbstractFloat, T <: Real}

    #Initialize values
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    penalty = copy(penalty0)
    
    #Solver for first value
    astate, pM, pU, iter = penalized_likelihood_hungarian(pM0, pU0, compsum, priorM, priorU, penalty, tol = tol, maxIter = maxIter, cluster = cluster, verbose = verbose)
    
    #Write values
    outLinks = [astate.nassigned]
    outM = reshape(pM, (length(pM), 1))
    outU = reshape(pU, (length(pM), 1))
    outIter = [iter]
    penalties = [penalty0]

    #Inititalize C trace vars
    outrows, outcols, outstart, outstop = Int[], Int[], Int[], Int[]
    prevrow2col = copy(astate.r2c)
    startrow2col = ones(Int, astate.nrow) #zeros(Int, length(currrow2col))
    
    ii = 1

    #Increment Penalty
    penalty, nabove = incr_penalty(astate.r2c, pM, pU, compsum, penalty, mininc)
    w = penalized_weights_vector(pM, pU, compsum, penalty)
    pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU, w, 0.0) #updates removing links that are below new penalty
    
    while nabove > 0
        if verbose
            println("penalty: $penalty, matches: $(astate.nassigned), nabove: $nabove")
        end
        if logflag
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "penalty: $penalty, matches: $(astate.nassigned), nabove: $nabove\n\n")
            end
        end

        #Find MAP with new penalty
        astate, pM, pU, iter = penalized_likelihood_hungarian(pM, pU, compsum, priorM, priorU, penalty, tol = tol, maxIter = maxIter, cluster = cluster, verbose = verbose)

        #Write values
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        push!(outLinks, astate.nassigned)
        outIter = push!(outIter, iter)
        push!(penalties, penalty)

        #Update C trace vars
        ii += 1
        outrows, outcols, outstart, outstop, startrow2col = update_Ctrace_vars!(astate.r2c, prevrow2col, ii, outrows, outcols, outstart, outstop, startrow2col)
        prevrow2col = copy(astate.r2c)
        
        #Increment Penalty
        penalty, nabove = incr_penalty(astate.r2c, pM, pU, compsum, penalty, mininc)
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU, w, 0.0) #updates removing links that are below new penalty
    end

    #Add final C of matches to C trace value
    #ii += 1
    outrows, outcols, outstart, outstop, startrow2col = update_Ctrace_vars!(prevrow2col, ii, outrows, outcols, outstart, outstop, startrow2col)
    
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
    #Initialize values
    pseudoM = priorM - ones(T, length(priorM))
    pseudoU = priorU - ones(T, length(priorU))
    penalty = copy(penalty0)
        
    #Solver for first value
    astate, pM, pU, iter = penalized_likelihood_auction(pM0, pU0, compsum, priorM, priorU, penalty0, epsiscale = epsiscale, minmargin = minmargin, digt = digt, tol = tol, maxIter = maxIter, cluster = cluster, update = update, verbose = verbose)
    
    #Write values
    outLinks = [astate.nassigned]
    outM = reshape(pM, (length(pM), 1))
    outU = reshape(pU, (length(pM), 1))
    outIter = [iter]
    penalties = [penalty0]

    #Inititalize C trace vars
    outrows, outcols, outstart, outstop = Int[], Int[], Int[], Int[]
    prevrow2col = copy(astate.r2c)
    startrow2col = ones(Int, astate.nrow) #zeros(Int, length(currrow2col))
    
    ii = 1

    #Increment Penalty
    penalty, nabove = incr_penalty(astate.r2c, pM, pU, compsum, penalty, mininc)
    w = penalized_weights_vector(pM, pU, compsum, penalty)
    pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU, w, 0.0) #updates removing links that are below new penalty
        
    while nabove > 0
        if verbose
            println("penalty: $penalty, matches: $(astate.nassigned), nabove: $nabove")
        end
        if logflag
            open(logfile, "a") do f
                write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
                write(f, "penalty: $penalty, matches: $(astate.nassigned), nabove: $nabove\n\n")
            end
        end

        #Find MAP with new penalty
        astate, pM, pU, iter = penalized_likelihood_auction(pM, pU, compsum, priorM, priorU, penalty, epsiscale = epsiscale, minmargin = minmargin, digt = digt, tol = tol, maxIter = maxIter, cluster = cluster, update = update, verbose = verbose)

        #Write values
        outM = hcat(outM, pM)
        outU = hcat(outU, pU)
        push!(outLinks, astate.nassigned)
        outIter = push!(outIter, iter)
        push!(penalties, penalty)

        #Update C trace vars
        ii += 1
        outrows, outcols, outstart, outstop, startrow2col = update_Ctrace_vars!(astate.r2c, prevrow2col, ii, outrows, outcols, outstart, outstop, startrow2col)
        prevrow2col = copy(astate.r2c)
        
        
        #Increment Penalty
        penalty, nabove = incr_penalty(astate.r2c, pM, pU, compsum, penalty, mininc)
        w = penalized_weights_vector(pM, pU, compsum, penalty)
        pM, pU = max_MU(astate.r2c, compsum, pseudoM, pseudoU, w, 0.0) #updates removing links that are below new penalty
    end

    #Add final C of matches to C trace value
    #ii += 1
    outrows, outcols, outstart, outstop, startrow2col = update_Ctrace_vars!(prevrow2col, ii, outrows, outcols, outstart, outstop, startrow2col)
    
    return ParameterChain([outrows outcols outstart outstop], outLinks, permutedims(outM), permutedims(outU), length(outLinks), true), penalties, outIter
end
