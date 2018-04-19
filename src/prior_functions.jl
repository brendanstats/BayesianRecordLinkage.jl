####################
#Green and Mardia 2006
####################

exppenalty_prior(nlink::Integer, θ::Real) = exp(-θ * nlink)
exppenalty_prior(C::LinkMatrix, θ::Real) = exppenalty_prior(C.nlink, θ)

exppenalty_logprior(nlink::Integer, θ::Real) = -θ * nlink
exppenalty_logprior(C::LinkMatrix, θ::Real) = exppenalty_logprior(C.nlink, θ)

"""
p(C1) / P(C2)
"""
function exppenalty_ratio(nlink1::Integer, nlink2::Integer, θ::Real)
    if nlink1 == nlink2
        return 1.0
    else
        return exp(θ * (nlink2 - nlink1))
    end
end

function exppenalty_ratio(C1::LinkMatrix, C2::LinkMatrix, θ::Real)
    if (C1.nrow != C2.nrow) || (C1.ncol != C2.ncol)
        error("Dimensions of LinkMatrix1 and LinkMatrix2 must match")
    end
    return exppenalty_ratio(C1.nlink, C2.nlink, θ)
end

"""
P(C with 1 more link) / P(C)
"""
exppenalty_ratiop1(θ::Real) = exp(-θ)
exppenalty_ratiop1(nlink::Integer, θ::Real) = exp(-θ)

"""
P(C) / P(C with 1 less link)
"""
exppenalty_ratiom1(θ::Real) = exp(θ)
exppenalty_ratiom1(nlink::Integer, θ::Real) = exp(θ)

function exppenalty_logratio(nlink1::Integer, nlink2::Integer, θ::Real)
    if nlink1 == nlink2
        return 1.0
    else
        return θ * (nlink2 - nlink1))
    end
end

function exppenalty_logratio(C1::LinkMatrix, C2::LinkMatrix, θ::Real)
    if (C1.nrow != C2.nrow) || (C1.ncol != C2.ncol)
        error("Dimensions of LinkMatrix1 and LinkMatrix2 must match")
    end
    return exppenalty_logratio(C1.nlink, C2.nlink, θ)
end

"""
P(C with 1 more link) / P(C)
"""
exppenalty_logratiop1(θ::Real) = -θ
exppenalty_logratiop1(nlink::Integer, θ::Real) = -θ

"""
P(C) / P(C with 1 less link)
"""
exppenalty_logratiom1(θ::Real) = θ
exppenalty_logratiom1(nlink::Integer, θ::Real) = θ

####################
#Sadinle 2017
####################

function betabipartite_prior(nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    if nrow >= ncol
        return prod(nlink + 1:nrow) * beta(nlink + α, ncol - nlink + β) / beta(α, β)
    else
        return prod(nlink + 1:ncol) * beta(nlink + α, nrow - nlink + β) / beta(α, β)
    end
end

betabipartite_prior(C::LinkMatrix, α::Real, β::Real) = betabipartite_prior(nlink, nrow, ncol, α, β)

function betabipartite_logprior(nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    if nrow >= ncol
        return lfact(nrow - nlink) - lfact(nrow) +  lbeta(nlink + α, ncol - nlink + β) - lbeta(α, β)
    else
        return lfact(ncol - nlink) - lfact(ncol) + lbeta(nlink + α, nrow - nlink + β) - lbeta(α, β)
    end
end

betabipartite_logprior(C::LinkMatrix, α::Real, β::Real) = betabipartite_logprior(nlink, nrow, ncol, α, β)

function betabipartite_ratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    if nlink1 == nlink2
        return 1.0
    elseif nlink1 > nlink2 #adjust first term in ratio
        if nrow >= ncol
            return prod((nrow - nlink1):(nrow - nlink2))^-1 * beta(nlink1 + α, ncol - nlink1 + β) / beta(nlink2 + α, ncol - nlink2 + β)
        else
            return prod((ncol - nlink1):(ncol - nlink2))^-1 * beta(nlink1 + α, nrow - nlink1 + β) / beta(nlink2 + α, nrow - nlink2 + β)
        end
    else #nlink1 < nlink2
        if nrow >= ncol
            return prod((nrow - nlink2):(nrow - nlink1)) * beta(nlink1 + α, ncol - nlink1 + β) / beta(nlink2 + α, ncol - nlink2 + β)
        else
            return prod((ncol - nlink2):(ncol - nlink1)) * beta(nlink1 + α, nrow - nlink1 + β) / beta(nlink2 + α, nrow - nlink2 + β)
        end
    end
end

function betabipartite_ratio(C1::LinkMatrix, C2::LinkMatrix, α::Real, β::Real)
    if (C1.nrow != C2.nrow) || (C1.ncol != C2.ncol)
        error("Dimensions of LinkMatrix1 and LinkMatrix2 must match")
    end
    return betabipartite_ratio(C1.nlink, C2.nlink, C1.nrow, C1.ncol, α, β)
end
#    if C1.nlink == C2.nlink
#        return 1.0
#    elseif C1.nlink > C2.nlink #adjust first term in ratio
#        if C1.nrow >= C1.ncol
#            return prod((C1.nrow - C1.nlink):(C2.nrow - C2.nlink))^-1 * beta(C1.nlink + α, C1.ncol - C1.nlink + β) / beta(C2.nlink + α, C2.ncol - C2.nlink + β)
#        else
#            return prod((C1.ncol - C1.nlink):(C2.ncol - C2.nlink))^-1 * beta(C1.nlink + α, C1.nrow - C1.nlink + β) / beta(C2.nlink + α, C2.nrow - C2.nlink + β)
#        end
#    else #C1.nlink < C2.nlink
#        if C1.nrow >= C1.ncol
#            return prod((C2.nrow - C2.nlink):(C1.nrow - C1.nlink)) * beta(C1.nlink + α, C1.ncol - C1.nlink + β) / beta(C2.nlink + α, C2.ncol - C2.nlink + β)
#        else
#            return prod((C2.ncol - C2.nlink):(C1.ncol - C1.nlink)) * beta(C1.nlink + α, C1.nrow - C1.nlink + β) / beta(C2.nlink + α, C2.nrow - C2.nlink + β)
#        end
#    end
#end

function betabipartite_logratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    if nlink1 == nlink2
        return 0.0
    else
        if  nrow >= ncol
            return lfact(nrow - nlink1) - lfact(nrow - nlink2) + lbeta(nlink1 + α, ncol - nlink1 + β) - lbeta(nlink2 + α, ncol - nlink2 + β)
        else
            return lfact(ncol - nlink1) - lfact(ncol - nlink2) + lbeta(nlink1 + α, nrow - nlink1 + β) - lbeta(nlink2 + α, nrow - nlink2 + β)
        end
    end
end

function betabipartite_logratio(C1::LinkMatrix, C2::LinkMatrix, α::Real, β::Real)
    if (C1.nrow != C2.nrow) || (C1.ncol != C2.ncol)
        error("Dimensions of LinkMatrix1 and LinkMatrix2 must match")
    end
    return betabipartite_logratio(C1.nlink, C2.nlink, C1.nrow, C1.ncol, α, β)
end
#    if C1.nlink == C2.nlink
#        return 0.0
#    else
#        if  C1.nrow >= C1.ncol
#            return lfact(C1.nrow - C1.nlink) - lfact(C2.nrow - C2.nlink) + lbeta(C1.nlink + α, C1.ncol - C1.nlink + β) - lbeta(C2.nlink + α, C2.ncol - C2.nlink + β)
#        else
#            return lfact(C1.ncol - C1.nlink) - lfact(C2.ncol - C2.nlink) + lbeta(C1.nlink + α, C1.nrow - C1.nlink + β) - lbeta(C2.nlink + α, C2.nrow - C2.nlink + β)
#        end
#    end
#end
