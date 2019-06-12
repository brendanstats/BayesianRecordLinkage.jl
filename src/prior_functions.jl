####################
#Green and Mardia 2006 - these do not include normalizing constants, can computed by
#softmax(exppenalty_logprior.(0:min(nrow, ncol)))
####################

"""
    exppenalty_prior(nlink::Integer, θ::Real)
    exppenalty_prior(nlink::Integer, nrow::Integer, ncol::Integer, θ::Real)
    exppenalty_prior(C::LinkMatrix, θ::Real)

Prior which applies an exponential penalty based on the number of links returning exp(-nlink * θ)

Based on Green and Mardia "Bayesian alignment using hierarchical models, with applications in protein bioinformatics" (2007).
Return values is not normalized.

See also: [`exppenalty_logprior`](@ref), [`exppenalty_ratio`](@ref), [`exppenalty_logratio`](@ref), [`exppenalty_ratiopn`](@ref), [`exppenalty_logratiopn`](@ref)
"""
exppenalty_prior(nlink::Integer, θ::Real) = exp(-θ * nlink)
exppenalty_prior(nlink::Integer, nrow::Integer, ncol::Integer, θ::Real) = exp(-θ * nlink)
exppenalty_prior(C::LinkMatrix, θ::Real) = exppenalty_prior(C.nlink, θ)

"""
    exppenalty_logprior(nlink::Integer, θ::Real)
    exppenalty_logprior(nlink::Integer, nrow::Integer, ncol::Integer, θ::Real)
    exppenalty_logprior(C::LinkMatrix, θ::Real)

Prior which applies an exponential penalty based on the number of links returning exp(-nlink * θ), result in logspace.

Based on Green and Mardia "Bayesian alignment using hierarchical models, with applications in protein bioinformatics" (2007).
Return values is not normalized.

See also: [`exppenalty_prior`](@ref), [`exppenalty_ratio`](@ref), [`exppenalty_logratio`](@ref), [`exppenalty_ratiopn`](@ref), [`exppenalty_logratiopn`](@ref)
"""
exppenalty_logprior(nlink::Integer, θ::Real) = -θ * nlink
exppenalty_logprior(nlink::Integer, nrow::Integer, ncol::Integer, θ::Real) = -θ * nlink
exppenalty_logprior(C::LinkMatrix, θ::Real) = exppenalty_logprior(C.nlink, θ)

"""
    exppenalty_ratio(nlink1::Integer, nlink2::Integer, θ::Real)
    exppenalty_ratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, θ::Real)
    exppenalty_ratio(C1::LinkMatrix, C2::LinkMatrix, θ::Real)

Returns P(C1) / P(C2) for exponential penalty prior.

Based on Green and Mardia "Bayesian alignment using hierarchical models, with applications in protein bioinformatics" (2007).

See also: [`exppenalty_prior`](@ref), [`exppenalty_logprior`](@ref), [`exppenalty_logratio`](@ref), [`exppenalty_ratiopn`](@ref), [`exppenalty_logratiopn`](@ref)
"""
function exppenalty_ratio(nlink1::Integer, nlink2::Integer, θ::Real)
    if nlink1 == nlink2
        return 1.0
    else
        return exp(θ * (nlink2 - nlink1))
    end
end
exppenalty_ratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, θ::Real) = exppenalty_ratio(nlink1, nlink2, θ)

function exppenalty_ratio(C1::LinkMatrix, C2::LinkMatrix, θ::Real)
    if (C1.nrow != C2.nrow) || (C1.ncol != C2.ncol)
        error("Dimensions of LinkMatrix1 and LinkMatrix2 must match")
    end
    return exppenalty_ratio(C1.nlink, C2.nlink, θ)
end

"""
    exppenalty_logratio(nlink1::Integer, nlink2::Integer, θ::Real)
    exppenalty_logratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, θ::Real)
    exppenalty_logratio(C1::LinkMatrix, C2::LinkMatrix, θ::Real)

Returns log(P(C1) / P(C2)) for exponential penalty prior.

Based on Green and Mardia "Bayesian alignment using hierarchical models, with applications in protein bioinformatics" (2007).

See also: [`exppenalty_prior`](@ref), [`exppenalty_logprior`](@ref), [`exppenalty_ratio`](@ref), [`exppenalty_logratio`](@ref), [`exppenalty_ratiopn`](@ref), [`exppenalty_logratiopn`](@ref)
"""
function exppenalty_logratio(nlink1::Integer, nlink2::Integer, θ::Real)
    if nlink1 == nlink2
        return 1.0
    else
        return θ * (nlink2 - nlink1)
    end
end

exppenalty_logratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, θ::Real) = exppenalty_logratio(nlink1, nlink2, θ)

function exppenalty_logratio(C1::LinkMatrix, C2::LinkMatrix, θ::Real)
    if (C1.nrow != C2.nrow) || (C1.ncol != C2.ncol)
        error("Dimensions of LinkMatrix1 and LinkMatrix2 must match")
    end
    return exppenalty_logratio(C1.nlink, C2.nlink, θ)
end

"""
    exppenalty_ratiopn(nadd::Integer, nlink::Integer, θ::Real)
    exppenalty_ratiopn(nadd::Integer, nlink::Integer, nrow::Integer, ncol::Integer, θ::Real)
    exppenalty_ratiopn(nadd::Integer, C::LinkMatrix, θ::Real)

P(C with L + n links) / P(C with L links) for exponential penalty prior.

Based on Green and Mardia "Bayesian alignment using hierarchical models, with applications in protein bioinformatics" (2007).

See also: [`exppenalty_prior`](@ref), [`exppenalty_logprior`](@ref), [`exppenalty_ratio`](@ref), [`exppenalty_logratio`](@ref), [`exppenalty_ratiopn`](@ref), [`exppenalty_logratiopn`](@ref)
"""
exppenalty_ratiopn(nadd::Integer, nlink::Integer, θ::Real) = exp(-θ * nadd)
exppenalty_ratiopn(nadd::Integer, nlink::Integer, nrow::Integer, ncol::Integer, θ::Real) = exp(-θ * nadd)
exppenalty_ratiopn(nadd::Integer, C::LinkMatrix, θ::Real) = exp(-θ * nadd)

"""
    exppenalty_logratiopn(nadd::Integer, nlink::Integer, θ::Real)
    exppenalty_logratiopn(nadd::Integer, nlink::Integer, nrow::Integer, ncol::Integer, θ::Real)
    exppenalty_logratiopn(nadd::Integer, C::LinkMatrix, θ::Real)

log(P(C with L + n links) / P(C with L links)) for exponential penalty prior.

Based on Green and Mardia "Bayesian alignment using hierarchical models, with applications in protein bioinformatics" (2007)

See also: [`exppenalty_prior`](@ref), [`exppenalty_logprior`](@ref), [`exppenalty_ratio`](@ref), [`exppenalty_logratio`](@ref), [`exppenalty_ratiopn`](@ref), [`exppenalty_logratiopn`](@ref)
"""
exppenalty_logratiopn(nadd::Integer, nlink::Integer, θ::Real) = -θ * nadd
exppenalty_logratiopn(nadd::Integer, nlink::Integer, nrow::Integer, ncol::Integer, θ::Real) = -θ * nadd
exppenalty_logratiopn(nadd::Integer, C::LinkMatrix, θ::Real) = -θ * nadd

####################
#Sadinle 2017
####################

"""
    betabipartite_prior(nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    betabipartite_prior(C::LinkMatrix, α::Real, β::Real)

Density for beta distribution for bipartite matchings.

Terminology taken from Sadinle "Bayesian Estimation of Bipartite Matchings for Record Linkage" (2017).

See also: [`betabipartite_logprior`](@ref), [`betabipartite_ratio`](@ref), [`betabipartite_logratio`](@ref), [`betabipartite_ratiopn`](@ref), [`betabipartite_logratiopn`](@ref)
"""
function betabipartite_prior(nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    return nrow >= ncol ? prod(nlink + 1:nrow) * beta(nlink + α, ncol - nlink + β) / beta(α, β) : prod(nlink + 1:ncol) * beta(nlink + α, nrow - nlink + β) / beta(α, β)
end

betabipartite_prior(C::LinkMatrix, α::Real, β::Real) = betabipartite_prior(nlink, nrow, ncol, α, β)

"""
    betabipartite_logprior(nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    betabipartite_logprior(C::LinkMatrix, α::Real, β::Real)

Log density for beta distribution for bipartite matchings.

Terminology taken from Sadinle "Bayesian Estimation of Bipartite Matchings for Record Linkage" (2017).

See also: [`betabipartite_prior`](@ref), [`betabipartite_ratio`](@ref), [`betabipartite_logratio`](@ref), [`betabipartite_ratiopn`](@ref), [`betabipartite_logratiopn`](@ref)
"""
function betabipartite_logprior(nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    nrow >= ncol ? lfactorial(nrow - nlink) - lfactorial(nrow) +  lbeta(nlink + α, ncol - nlink + β) - lbeta(α, β) : lfactorial(ncol - nlink) - lfactorial(ncol) + lbeta(nlink + α, nrow - nlink + β) - lbeta(α, β)
end

betabipartite_logprior(C::LinkMatrix, α::Real, β::Real) = betabipartite_logprior(C.nlink, C.nrow, C.ncol, α, β)

"""
    betabipartite_ratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    betabipartite_ratio(C1::LinkMatrix, C2::LinkMatrix, α::Real, β::Real)

Ratio of beta distribution for bipartite matchings, equivalent to `betabipartite_prior(C1) / betabipartite_prior(C2)`.

Terminology taken from Sadinle "Bayesian Estimation of Bipartite Matchings for Record Linkage" (2017).

See also: [`betabipartite_prior`](@ref), [`betabipartite_logprior`](@ref), [`betabipartite_logratio`](@ref), [`betabipartite_ratiopn`](@ref), [`betabipartite_logratiopn`](@ref)
"""
function betabipartite_ratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    if nlink1 == nlink2
        return 1.0
    elseif nlink1 > nlink2 #adjust first term in ratio
        nrow >= ncol ? prod((nrow - nlink1):(nrow - nlink2))^-1 * beta(nlink1 + α, ncol - nlink1 + β) / beta(nlink2 + α, ncol - nlink2 + β) : prod((ncol - nlink1):(ncol - nlink2))^-1 * beta(nlink1 + α, nrow - nlink1 + β) / beta(nlink2 + α, nrow - nlink2 + β)
    else #nlink1 < nlink2
        nrow >= ncol ? prod((nrow - nlink2):(nrow - nlink1)) * beta(nlink1 + α, ncol - nlink1 + β) / beta(nlink2 + α, ncol - nlink2 + β) : prod((ncol - nlink2):(ncol - nlink1)) * beta(nlink1 + α, nrow - nlink1 + β) / beta(nlink2 + α, nrow - nlink2 + β)
    end
end

function betabipartite_ratio(C1::LinkMatrix, C2::LinkMatrix, α::Real, β::Real)
    if (C1.nrow != C2.nrow) || (C1.ncol != C2.ncol)
        error("Dimensions of LinkMatrix1 and LinkMatrix2 must match")
    end
    return betabipartite_ratio(C1.nlink, C2.nlink, C1.nrow, C1.ncol, α, β)
end

"""
    betabipartite_ratio(nadd::Integer, nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    betabipartite_ratio(C1::LinkMatrix, C::LinkMatrix, α::Real, β::Real)

Ratio of beta distribution for bipartite matchings, equivalent to `betabipartite_prior(C1) / betabipartite_prior(C2)`.

Terminology taken from Sadinle "Bayesian Estimation of Bipartite Matchings for Record Linkage" (2017).

See also: [`betabipartite_prior`](@ref), [`betabipartite_logprior`](@ref), [`betabipartite_ratio`](@ref), [`betabipartite_logratio`](@ref), [`betabipartite_logratiopn`](@ref)
"""
betabipartite_ratiopn(nadd::Integer, nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real) = exp(betabipartite_logratiopn(nadd, nlink, nrow, ncol, α, β))
betabipartite_ratiopn(nadd::Integer, C::LinkMatrix, α::Real, β::Real) = exp(betabipartite_logratiopn(nadd, C.nlink, C.nrow, C.ncol, α, β))

"""
    betabipartite_logratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    betabipartite_logratio(C1::LinkMatrix, C2::LinkMatrix, α::Real, β::Real)

Log ratio of beta distribution for bipartite matchings densities, equivalent to `betabipartite_logprior(C1) - betabipartite_logprior(C2)`.

Terminology taken from Sadinle "Bayesian Estimation of Bipartite Matchings for Record Linkage" (2017).

See also: [`betabipartite_prior`](@ref), [`betabipartite_logprior`](@ref), [`betabipartite_ratio`](@ref), [`betabipartite_ratiopn`](@ref), [`betabipartite_logratiopn`](@ref)
"""
function betabipartite_logratio(nlink1::Integer, nlink2::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    if nlink1 == nlink2
        return 0.0
    else
        nrow >= ncol ? lfactorial(nrow - nlink1) - lfactorial(nrow - nlink2) + lbeta(nlink1 + α, ncol - nlink1 + β) - lbeta(nlink2 + α, ncol - nlink2 + β) : lfactorial(ncol - nlink1) - lfactorial(ncol - nlink2) + lbeta(nlink1 + α, nrow - nlink1 + β) - lbeta(nlink2 + α, nrow - nlink2 + β)
    end
end

function betabipartite_logratio(C1::LinkMatrix, C2::LinkMatrix, α::Real, β::Real)
    if (C1.nrow != C2.nrow) || (C1.ncol != C2.ncol)
        error("Dimensions of LinkMatrix1 and LinkMatrix2 must match")
    end
    return betabipartite_logratio(C1.nlink, C2.nlink, C1.nrow, C1.ncol, α, β)
end

"""
    betabipartite_logratiopn(nadd::Integer, nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    betabipartite_logratiopn(nadd::Integer, C::LinkMatrix, α::Real, β::Real)

Log ratio of beta distribution for bipartite matchings densities, equivalent to `betabipartite_logprior(C with nadd more links) - betabipartite_logprior(C)`.

Terminology taken from Sadinle "Bayesian Estimation of Bipartite Matchings for Record Linkage" (2017).

See also: [`betabipartite_prior`](@ref), [`betabipartite_logprior`](@ref), [`betabipartite_ratio`](@ref), [`betabipartite_logratio`](@ref), [`betabipartite_ratiopn`](@ref)
"""
function betabipartite_logratiopn(nadd::Integer, nlink::Integer, nrow::Integer, ncol::Integer, α::Real, β::Real)
    if nadd == 0
        return 0.0
    end
    if nrow < ncol
        return betabipartite_logratiopn(nadd, nlink, ncol, nrow, α, β)
    end
    nnew = nlink + nadd
    coeff1 = lfactorial(nrow - nnew) - lfactorial(nrow - nlink)
    coeff2 = lbeta(nnew + α, ncol - nnew + β) - lbeta(nlink + α, ncol - nlink + β)
    return coeff1 + coeff2
end

betabipartite_logratiopn(nadd::Integer, C::LinkMatrix, α::Real, β::Real) = betabipartite_logratiopn(nadd, C.nlink, C.nrow, C.ncol, α, β)
