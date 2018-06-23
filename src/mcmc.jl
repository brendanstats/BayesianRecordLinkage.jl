"""
Compute the conditional dirchlet distribution for updating M and U parameters via a gibbs step
"""
function dirichlet_draw(matchcounts::Array{<:Integer, 1},
                            compsum::Union{ComparisonSummary, SparseComparisonSummary},
                            priorM::Array{<: Real, 1} = zeros(Float64, length(matchcounts)),
                            priorU::Array{<: Real, 1} = zeros(Float64, length(matchcounts)))
    nonmatchcounts = compsum.counts - matchcounts
    paramM = matchcounts + priorM
    paramU = nonmatchcounts + priorU

    pM = Array{Float64}(length(priorM))
    pU = Array{Float64}(length(priorU))
    
    startidx = 1
    for ii in 1:length(compsum.nlevels)
        rng = range(startidx, compsum.nlevels[ii])
        startidx += compsum.nlevels[ii]
        pM[rng] = rand(Dirichlet(paramM[rng]))
        pU[rng] = rand(Dirichlet(paramU[rng]))
    end
    return pM, pU
end

"""
Add Gibbs Step for M and U probabilities assuming a Dirchlet Prior
"""
function mh_gibbs_chain{T <: Real}(
    outerIter::Integer,
    C0::LinkMatrix,
    compsum::ComparisonSummary,
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpdfC::Function,
    transitionC::Function;
    innerIter::Integer = 1,
    informedMoves::Bool = true,
    ratioPrior::Bool = true,
    gibbsInner::Bool = true,
    sparseLinks::Bool = true)
    
    #MCMC Chains
    if sparseLinks
        CArray = spzeros(Int64, C0.nrow, outerIter)
    else
        CArray = zeros(Int64, C0.nrow, outerIter)
    end
    nlinkArray = Array{Int64}(outerIter)
    MArray = Array{Float64}(length(priorM), outerIter)
    UArray = Array{Float64}(length(priorU), outerIter)
    transC = zeros(Int64, outerIter)

    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    obsDeltas = obs_delta(compsum)
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    #loglikMargin = countDeltas' * logDiff, check this matrix multiplication
    
    #Outer iteration (recorded)
    for ii in 1:outerIter

        #Inner iteration
        for jj in innerIter
            
            if informedMoves
                propC, countdelta, ratioC = transitionC(C, compsum, countDeltas, logDiff, logpdfC, ratioPrior)
                #propC, countdelta, marginLoglik, ratioC = transitionC(C, compsum, countDeltas, loglikMargin, logpdfC, ratioPrior)
            else
                propC, countdelta, ratioC = transitionC(C, compsum, countDeltas)
                #propC, countdelta, marginLoglik, ratioC = transitionC(C, compsum, loglikeMargin)
            end
            
            #compute likelihood ratios
            if ratioPrior
                ratio = exp(logpdfC(propC, C) + dot(countdelta, logDiff)) * ratioC
                #ratio = exp(logpdfC(propC, C) + marginLoglik) * ratioC
            else
                ratio = exp(logpdfC(propC) - logpdfC(C) + dot(countdelta, logDiff)) * ratioC
                #ratio = exp(logpdfC(propC) - logpdfC(C) + marginLogLik) * ratioC
            end

            ##Accept move with probability min(1.0, ratio)
            if rand() < ratio

                ##Check that move has actually occured
                if C != propC
                    transC[ii] += 1
                    C = propC
                    matchcounts += countdelta
                end 
            end

            ##Perform Gibbs update if performed with inner iterations
            if gibbsInner
                pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
                logDiff = log.(pM) - log.(pU)
                #loglikMargin = countDeltas' * logDiff, check this matrix multiplication
            end
        end

        ##Perform Gibbs update if performed with outer iterations
        if !gibbsInner
            pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
            logDiff = log.(pM) - log.(pU)
            #loglikMargin = countDeltas' * logDiff, check this matrix multiplication
        end
        
        #Add states to chain
        if sparseLinks
            CArray[:, ii] = C.row2col
        else
            CArray[:, ii] = full(C.row2col)
        end
        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end
    return CArray', nlinkArray, MArray', UArray', transC
end

#@code_warntype
#view()
#, G <: Integer
function mh_gibbs_chain{T <: Real}(
    outerIter::Integer,
    C0::LinkMatrix,
    blockRanges::Array{CartesianRange{CartesianIndex{2}}, 1},
    compsum::Union{ComparisonSummary, SparseComparisonSummary},
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpdfC::Function,
    transitionC::Function;
    innerIter::Integer = 1,
    informedMoves::Bool = true,
    ratioPrior::Bool = true,
    gibbsInner::Bool = true,
    sparseLinks::Bool = true)
    
    #MCMC Chains
    if sparseLinks
        CArray = spzeros(Int64, C0.nrow, outerIter)
    else
        CArray = zeros(Int64, C0.nrow, outerIter)
    end
    nblocks = length(blockRanges)
    blockRows = map(x -> size(x)[1], blockRanges)
    blockCols = map(x -> size(x)[2], blockRanges)
    nlinkArray = Array{Int64}(outerIter)
    MArray = Array{Float64}(length(priorM), outerIter)
    UArray = Array{Float64}(length(priorU), outerIter)
    transC = zeros(Int64, outerIter, nblocks)
    
    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    obsDeltas = obs_delta(compsum)
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    #loglikMargin = countDeltas' * logDiff, check this matrix multiplication
    
    #Outer iteration (recorded)
    for ii in 1:outerIter

        #Inner iteration
        for jj in innerIter

            #Loop over blocks
            for kk in 1:nblocks
                #if blockRows[kk] == 1
                #    if blockCols[kk] = 1
                #    else
                #    end
                #elseif blockCols[kk] = 1
                #elseif blockRows[kk] == 2 && blockCols[kk] == 2
                #end
                #if blockRows[kk] == 1 && blockCols[kk] == 1
                #    if rand() < logistic(dot(countDeltas[:, compsum.obsidx[blockRows[kk].start]], logDiff))
                #       if !has_link(blockRows[kk].start..., C)
                #           add_link!(blockRows[kk].start..., C)
                #       end
                #   else
                #       if has_link(blockRows[kk].start..., C)
                #           remove_link!(blockRows[kk].start..., C)
                #       end
                #   end
                #end
                
                if informedMoves
                    propC, countdelta, ratioC = transitionC(blockRanges[kk], C, compsum, countDeltas, logDiff, logpdfC, ratioPrior)
                    #propC, countdelta, marginLoglik, ratioC = transitionC(blockRanges[kk], C, compsum, countDeltas, loglikMargin, logpdfC, ratioPrior)
                else
                    propC, countdelta, ratioC = transitionC(blockRanges[kk], C, compsum, countDeltas)
                    #propC, countdelta, marginLoglik, ratioC = transitionC(blockRanges[kk], C, compsum, loglikeMargin)
                end
                
                #compute likelihood ratios
                if ratioPrior
                    ratio = exp(logpdfC(propC, C) + dot(countdelta, logDiff)) * ratioC
                    #ratio = exp(logpdfC(propC, C) + marginLoglik) * ratioC
                else
                    ratio = exp(logpdfC(propC) - logpdfC(C) + dot(countdelta, logDiff)) * ratioC
                    #ratio = exp(logpdfC(propC) - logpdfC(C) + marginLogLik) * ratioC
                end

                ##Accept move with probability min(1.0, ratio)
                if rand() < ratio
                    
                    ##Check that move has actually occured
                    if C != propC
                        transC[ii, kk] += 1
                        C = propC
                        matchcounts += countdelta
                    end 
                end
            end #end block loop

            ##Perform Gibbs update if performed with inner iterations
            if gibbsInner
                pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
                logDiff = log.(pM) - log.(pU)
            end
        end

        ##Perform Gibbs update if performed with outer iterations
        if !gibbsInner
            pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
            logDiff = log.(pM) - log.(pU)
        end
        
        #Add states to chain
        if sparseLinks
            CArray[:, ii] = C.row2col
        else
            CArray[:, ii] = full(C.row2col)
        end
        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end
    return CArray', nlinkArray, MArray', UArray', transC
end

function mh_gibbs_count{T <: Real}(
    outerIter::Integer,
    C0::LinkMatrix,
    blockRanges::Array{CartesianRange{CartesianIndex{2}}, 1},
    compsum::Union{ComparisonSummary, SparseComparisonSummary},
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpdfC::Function,
    transitionC::Function;
    innerIter::Integer = 1,
    informedMoves::Bool = true,
    ratioPrior::Bool = true,
    gibbsInner::Bool = true,
    sparseLinks::Bool = true)
    
    #MCMC Chains
    if sparseLinks
        CArray = spzeros(Int64, C0.nrow, C0.ncol)
    else
        CArray = zeros(Int64, C0.nrow, C0.ncol)
    end
    nblocks = length(blockRanges)
    blockRows = map(x -> size(x)[1], blockRanges)
    blockCols = map(x -> size(x)[2], blockRanges)
    nlinkArray = Array{Int64}(outerIter)
    MArray = Array{Float64}(length(priorM), outerIter)
    UArray = Array{Float64}(length(priorU), outerIter)
    transC = zeros(Int64, nblocks)
    
    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    obsDeltas = obs_delta(compsum)
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    #loglikMargin = countDeltas' * logDiff, check this matrix multiplication
    
    #Outer iteration (recorded)
    for ii in 1:outerIter

        #Inner iteration
        for jj in innerIter

            #Loop over blocks
            for kk in 1:nblocks
                countdelta = countDeltas[:, compsum.obsidx[blockRanges[kk].start]]
                if blockRows[kk] == 1 && blockCols[kk] == 1
                    if iszero(C.row2col[blockRanges[kk].start[1]])
                        propC = add_link(blockRanges[kk].start.I..., C)
                        if ratioPrior
                            lp1 = dot(logDiff, countdelta) + logpdfC(propC, C)
                            if rand() <= logistic(lp1)
                                C = propC
                                transC[ii] += 1
                                matchcounts += countdelta #increase if adding
                            end
                        else
                            propC = remove_link(blockRanges[kk].start.I..., C)
                            lp1 = dot(logDiff, countdelta) + logpdfC(C, propC)
                            if rand() > logistic(lp1)
                                C = propC
                                transC[ii] += 1
                                matchcounts -= countdelta #delete if subtracting
                            end
                        end
                    else
                    end
                    continue
                end
                #if blockRows[kk] == 1
                #    if blockCols[kk] = 1
                #    else
                #    end
                #elseif blockCols[kk] = 1
                #elseif blockRows[kk] == 2 && blockCols[kk] == 2
                #end
                #if blockRows[kk] == 1 && blockCols[kk] == 1
                #    if rand() < logistic(dot(countDeltas[:, compsum.obsidx[blockRows[kk].start]], logDiff))
                #       if !has_link(blockRows[kk].start..., C)
                #           add_link!(blockRows[kk].start..., C)
                #       end
                #   else
                #       if has_link(blockRows[kk].start..., C)
                #           remove_link!(blockRows[kk].start..., C)
                #       end
                #   end
                #end
                
                if informedMoves
                    propC, countdelta, ratioC = transitionC(blockRanges[kk], C, compsum, countDeltas, logDiff, logpdfC, ratioPrior)
                    #propC, countdelta, marginLoglik, ratioC = transitionC(blockRanges[kk], C, compsum, countDeltas, loglikMargin, logpdfC, ratioPrior)
                else
                    propC, countdelta, ratioC = transitionC(blockRanges[kk], C, compsum, countDeltas)
                    #propC, countdelta, marginLoglik, ratioC = transitionC(blockRanges[kk], C, compsum, loglikeMargin)
                end
                
                #compute likelihood ratios
                if ratioPrior
                    ratio = exp(logpdfC(propC, C) + dot(countdelta, logDiff)) * ratioC
                    #ratio = exp(logpdfC(propC, C) + marginLoglik) * ratioC
                else
                    ratio = exp(logpdfC(propC) - logpdfC(C) + dot(countdelta, logDiff)) * ratioC
                    #ratio = exp(logpdfC(propC) - logpdfC(C) + marginLogLik) * ratioC
                end

                ##Accept move with probability min(1.0, ratio)
                if rand() < ratio
                    
                    ##Check that move has actually occured
                    if C != propC
                        transC[kk] += 1
                        C = propC
                        matchcounts += countdelta
                    end 
                end
            end #end block loop

            ##Perform Gibbs update if performed with inner iterations
            if gibbsInner
                pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
                logDiff = log.(pM) - log.(pU)
            end
        end

        ##Perform Gibbs update if performed with outer iterations
        if !gibbsInner
            pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
            logDiff = log.(pM) - log.(pU)
        end
        
        #Add states to chain
        if sparseLinks
            for (row, col) in zip(findnz(C.row2col)...)
                CArray[row, col] += 1
            end
        else
            for row in 1:compsum.nrow
                if !izero(C.row2col[row])
                    CArray[row, row2col[row]] += 1
                end
            end
        end
        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end
    return CArray, nlinkArray, MArray', UArray', transC, C
end

function mh_gibbs_chain_inplace{T <: Real}(
    outerIter::Integer,
    C0::LinkMatrix,
    compsum::Union{ComparisonSummary, SparseComparisonSummary},
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpdfC::Function,
    transitionC!::Function;
    innerIter::Integer = 1,
    gibbsInner::Bool = true,
    sparseLinks::Bool = true)
    
    #MCMC Chains
    if sparseLinks
        CArray = spzeros(Int64, C0.nrow, outerIter)
    else
        CArray = zeros(Int64, C0.nrow, outerIter)
    end
    
    #CArray = Dict{Int64,Array{Int64,2}}()
    nlinkArray = Array{Int64}(outerIter)
    MArray = Array{Float64}(length(priorM), outerIter)
    UArray = Array{Float64}(length(priorU), outerIter)
    transC = 0
    
    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    obsDeltas = obs_delta(compsum)
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    loglikMargin = countDeltas' * logDiff
    
    #Outer iteration (recorded)
    for ii in 1:outerIter

        #Inner iteration
        for jj in innerIter
                
            countdelta, move = transitionC!(C, compsum, loglikMargin, countDeltas, logpdfC)

            if move
                transC += 1
                matchcounts += countdelta
            end

            ##Perform Gibbs update if performed with inner iterations
            if gibbsInner
                pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
                logDiff = log.(pM) - log.(pU)
                loglikMargin = countDeltas' * logDiff
            end
        end

        ##Perform Gibbs update if performed with outer iterations
        if !gibbsInner
            pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
            logDiff = log.(pM) - log.(pU)
            loglikMargin = countDeltas' * logDiff
        end
        
        #Add states to chain
        if sparseLinks
            CArray[:, ii] = C.row2col
        else
            CArray[:, ii] = full(C.row2col)
        end
        
        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end
    return CArray, nlinkArray, MArray', UArray', transC, C
end

function mh_gibbs_chain_inplace{T <: Real}(
    outerIter::Integer,
    C0::LinkMatrix,
    blockRanges::Array{CartesianRange{CartesianIndex{2}}, 1},
    compsum::Union{ComparisonSummary, SparseComparisonSummary},
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpdfC::Function,
    transitionC!::Function;
    innerIter::Integer = 1,
    gibbsInner::Bool = true,
    sparseLinks::Bool = true)

    #priorType::String = "base"
    
    #MCMC Chains
    if sparseLinks
        CArray = spzeros(Int64, C0.nrow, outerIter)
    else
        CArray = zeros(Int64, C0.nrow, outerIter)
    end
    
    nblocks = length(blockRanges)
    blockRows = map(x -> size(x)[1], blockRanges)
    blockCols = map(x -> size(x)[2], blockRanges)
    nlinkArray = Array{Int64}(outerIter)
    MArray = Array{Float64}(length(priorM), outerIter)
    UArray = Array{Float64}(length(priorU), outerIter)
    transC = zeros(Int64, nblocks)
    
    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    obsDeltas = obs_delta(compsum)
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    loglikMargin = countDeltas' * logDiff
    
    #Outer iteration (recorded)
    for ii in 1:outerIter

        #Inner iteration
        for jj in innerIter

            #Loop over blocks
            for kk in 1:nblocks
                move = false
                countdelta = zeros(matchcounts)
                if (blockRows[kk] == 1) && (blockCols[kk] == 1)
                    countdelta, move = singleton_gibbs!(blockRanges[kk], C, compsum, loglikMargin, countDeltas, logpdfC)
                else
                    countdelta, move = transitionC!(blockRanges[kk], C, compsum, loglikMargin, countDeltas, logpdfC)
                end
                if move
                    transC[kk] += 1
                    matchcounts += countdelta
                end
            end #end block loop

            ##Perform Gibbs update if performed with inner iterations
            if gibbsInner
                pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
                logDiff = log.(pM) - log.(pU)
                loglikMargin = countDeltas' * logDiff
            end
        end

        ##Perform Gibbs update if performed with outer iterations
        if !gibbsInner
            pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
            logDiff = log.(pM) - log.(pU)
            loglikMargin = countDeltas' * logDiff
        end

        #Add states to chain
        if sparseLinks
            CArray[:, ii] = C.row2col
        else
            CArray[:, ii] = full(C.row2col)
        end
        
        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end
    return CArray, nlinkArray, MArray', UArray', transC, C
end

function mh_gibbs_count_inplace{T <: Real}(
    outerIter::Integer,
    C0::LinkMatrix,
    compsum::Union{ComparisonSummary, SparseComparisonSummary},
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpdfC::Function,
    transitionC!::Function;
    innerIter::Integer = 1,
    gibbsInner::Bool = true,
    sparseLinks::Bool = true)
    
    #MCMC Chains
    if sparseLinks
        CArray = spzeros(Int64, C0.nrow, C0.ncol)
    else
        CArray = zeros(Int64, C0.nrow, C0.ncol)
    end
    nlinkArray = Array{Int64}(outerIter)
    MArray = Array{Float64}(length(priorM), outerIter)
    UArray = Array{Float64}(length(priorU), outerIter)
    transC = 0
    
    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    obsDeltas = obs_delta(compsum)
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    loglikMargin = countDeltas' * logDiff
    
    #Outer iteration (recorded)
    for ii in 1:outerIter

        #Inner iteration
        for jj in innerIter
                
            countdelta, move = transitionC!(C, compsum, loglikMargin, countDeltas, logpdfC)

            if move
                transC += 1
                matchcounts += countdelta
            end

            ##Perform Gibbs update if performed with inner iterations
            if gibbsInner
                pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
                logDiff = log.(pM) - log.(pU)
                loglikMargin = countDeltas' * logDiff
            end
        end

        ##Perform Gibbs update if performed with outer iterations
        if !gibbsInner
            pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
            logDiff = log.(pM) - log.(pU)
            loglikMargin = countDeltas' * logDiff
        end
        
        #Add states to chain
        if sparseLinks
            for (row, col) in zip(findnz(C.row2col)...)
                CArray[row, col] += 1
            end
        else
            for row in 1:compsum.nrow
                if !izero(C.row2col[row])
                    CArray[row, row2col[row]] += 1
                end
            end
        end
        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end
    return CArray, nlinkArray, MArray', UArray', transC, C
end

function mh_gibbs_count_inplace{T <: Real}(
    outerIter::Integer,
    C0::LinkMatrix,
    blockRanges::Array{CartesianRange{CartesianIndex{2}}, 1},
    compsum::Union{ComparisonSummary, SparseComparisonSummary},
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpdfC::Function,
    transitionC!::Function;
    innerIter::Integer = 1,
    gibbsInner::Bool = true,
    sparseLinks::Bool = true)

    #priorType::String = "base"
    
    #MCMC Chains
    if sparseLinks
        CArray = spzeros(Int64, C0.nrow, C0.ncol)
    else
        CArray = zeros(Int64, C0.nrow, C0.ncol)
    end
    nblocks = length(blockRanges)
    blockRows = map(x -> size(x)[1], blockRanges)
    blockCols = map(x -> size(x)[2], blockRanges)
    nlinkArray = Array{Int64}(outerIter)
    MArray = Array{Float64}(length(priorM), outerIter)
    UArray = Array{Float64}(length(priorU), outerIter)
    transC = zeros(Int64, nblocks)
    
    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    obsDeltas = obs_delta(compsum)
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    loglikMargin = countDeltas' * logDiff
    
    #Outer iteration (recorded)
    for ii in 1:outerIter

        #Inner iteration
        for jj in innerIter

            #Loop over blocks
            for kk in 1:nblocks
                move = false
                countdelta = zeros(matchcounts)
                if (blockRows[kk] == 1) && (blockCols[kk] == 1)
                    countdelta, move = singleton_gibbs!(blockRanges[kk], C, compsum, loglikMargin, countDeltas, logpdfC)
                else
                    countdelta, move = transitionC!(blockRanges[kk], C, compsum, loglikMargin, countDeltas, logpdfC)
                end
                if move
                    transC[kk] += 1
                    matchcounts += countdelta
                end
            end #end block loop

            ##Perform Gibbs update if performed with inner iterations
            if gibbsInner
                pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
                logDiff = log.(pM) - log.(pU)
                loglikMargin = countDeltas' * logDiff
            end
        end

        ##Perform Gibbs update if performed with outer iterations
        if !gibbsInner
            pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
            logDiff = log.(pM) - log.(pU)
            loglikMargin = countDeltas' * logDiff
        end

        #Add states to chain
        if sparseLinks
            for (row, col) in zip(findnz(C.row2col)...)
                CArray[row, col] += 1
            end
        else
            for row in 1:compsum.nrow
                if !izero(C.row2col[row])
                    CArray[row, row2col[row]] += 1
                end
            end
        end

        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end
    return CArray, nlinkArray, MArray', UArray', transC, C
end

function mh_gibbs_count_inplace{T <: Real}(
    outerIter::Integer,
    C0::LinkMatrix,
    blockRanges::Array{CartesianRange{CartesianIndex{2}}, 1},
    compsum::Union{ComparisonSummary, SparseComparisonSummary},
    priorM::Array{T, 1},
    priorU::Array{T, 1},
    logpdfC::Function,
    transitionC!::Function,
    rowtype::Array{<:Integer, 1},
    coltype::Array{<:Integer, 1};
    innerIter::Integer = 1,
    gibbsInner::Bool = true,
    sparseLinks::Bool = true)

    #priorType::String = "base"
    
    #MCMC Chains
    if sparseLinks
        CArray = spzeros(Int64, C0.nrow, C0.ncol)
    else
        CArray = zeros(Int64, C0.nrow, C0.ncol)
    end
    nblocks = length(blockRanges)
    blockRows = map(x -> size(x)[1], blockRanges)
    blockCols = map(x -> size(x)[2], blockRanges)
    nlinkArray = Array{Int64}(outerIter)
    MArray = Array{Float64}(length(priorM), outerIter)
    UArray = Array{Float64}(length(priorU), outerIter)
    transC = zeros(Int64, nblocks)

    ##Add counting for row and column type
    rowdim = maximum(rowtype) + 1
    coldim = maximum(coltype) + 1
    ntypesArray = zeros(Int64, rowdim * coldim, outerIter)
    
    ##Initial States
    countDeltas = counts_delta(compsum) #each column is an observation
    obsDeltas = obs_delta(compsum)
    C = deepcopy(C0)
    matchcounts, matchobs = counts_matches(C, compsum)
    pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
    logDiff = log.(pM) - log.(pU)
    loglikMargin = countDeltas' * logDiff
    
    #Outer iteration (recorded)
    for ii in 1:outerIter

        #Inner iteration
        for jj in innerIter

            #Loop over blocks
            for kk in 1:nblocks
                move = false
                countdelta = zeros(matchcounts)
                if (blockRows[kk] == 1) && (blockCols[kk] == 1)
                    countdelta, move = singleton_gibbs!(blockRanges[kk], C, compsum, loglikMargin, countDeltas, logpdfC)
                else
                    countdelta, move = transitionC!(blockRanges[kk], C, compsum, loglikMargin, countDeltas, logpdfC)
                end
                if move
                    transC[kk] += 1
                    matchcounts += countdelta
                end
            end #end block loop

            ##Perform Gibbs update if performed with inner iterations
            if gibbsInner
                pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
                logDiff = log.(pM) - log.(pU)
                loglikMargin = countDeltas' * logDiff
            end
        end

        ##Perform Gibbs update if performed with outer iterations
        if !gibbsInner
            pM, pU = dirichlet_draw(matchcounts, compsum, priorM, priorU)
            logDiff = log.(pM) - log.(pU)
            loglikMargin = countDeltas' * logDiff
        end

        #Add states to chain
        if sparseLinks
            for (row, col) in zip(findnz(C.row2col)...)
                CArray[row, col] += 1
                ntypesArray[sub2ind((rowdim, coldim), rowtype[row] + 1, coltype[col] + 1), ii] += 1
            end
        else
            for row in 1:compsum.nrow
                if !izero(C.row2col[row])
                    CArray[row, row2col[row]] += 1
                    ntypesArray[sub2ind((rowdim, coldim), rowtype[row] + 1, coltype[row2col[row]] + 1), ii] += 1
                end
            end
        end
        
        nlinkArray[ii] = C.nlink
        MArray[:, ii] = pM
        UArray[:, ii] = pU
    end
    return CArray, nlinkArray, ntypesArray', MArray', UArray', transC, C
end
