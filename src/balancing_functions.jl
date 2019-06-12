"""
    lsqrt(logx::AbstractFloat)

Compute log(sqrt(x)) fomr log(x).

See also: [`lbarker`](@ref), [`lmin1`](@ref), [`lmax1`](@ref), [`sqrt_logx`](@ref), [`sqrt`](@ref)
"""
lsqrt(logx::AbstractFloat) = 0.5 * logx


"""
    sqrt_logx(logx::AbstractFloat) = exp(0.5 * logx)

Compute sqrt(x) from log(x).

See also: [`lbarker_logx`](@ref), [`lmin1_logx`](@ref), [`lmax1_logx`](@ref), [`lsqrt`](@ref), [`sqrt`](@ref)
"""
sqrt_logx(logx::AbstractFloat) = exp(lsqrt(logx))

"""
    lbarker(logx::AbstractFloat)

Compute log(x / (1 + x)) from log(x).

See also: [`lsqrt`](@ref), [`lmin1`](@ref), [`lmax1`](@ref), [`barker_logx`](@ref), [`barker`](@ref)
"""
lbarker(logx::AbstractFloat) = logx - log1pexp(logx)

"""
    barker_logx(logx::AbstractFloat)

Compute x / (1 + x) from log(x), equivalent to the logistic function.

See also: [`barker_logx`](@ref), [`min1_logx`](@ref), [`max1_logx`](@ref), [`lbarker`](@ref), [`barker`](@ref)
"""
barker_logx(logx::AbstractFloat) = logistic(logx)

"""
    barker(x::T) where T <: Real

Compute x / (1 + x)

See also: [`sqrt`](@ref), [`min1`](@ref), [`max1`](@ref), [`lbarker`](@ref), [`barker_logx`](@ref)
"""
barker(x::T) where T <: Real = x / (x + one(T))

"""
    lmin1(logx::Real) where T <: Real

Compute log(min(x, 1)) from log(x) in logspace.

See also: [`lsqrt`](@ref), [`lbarker`](@ref), [`lmax1`](@ref), [`min1_logx`](@ref), [`min1`](@ref)
"""
lmin1(logx::Real) where T <: Real = min(logx, zero(T))

"""
    min1_logx(logx::T) where T <: AbstractFloat

Compute min(x, 1) from log(x).

See also: [`barker_logx`](@ref), [`min1_logx`](@ref), [`max1_logx`](@ref), [`lmin1`](@ref), [`min1`](@ref)
"""
min1_logx(logx::T) where T <: AbstractFloat = logx > zero(T) ? one(T) : exp(logx)

"""
    min1(x::Real) where T <: Real

Compute min(x, 1)

See also: [`sqrt`](@ref), [`barker`](@ref), [`max1`](@ref), [`lmin1`](@ref), [`min1_logx`](@ref)
"""
min1(x::Real) where T <: Real = min(x, one(T))

"""
    lmax1(logx::Real) where T <: Real

Compute log(max(x, 1)) from log(x) in logspace.

See also: [`lsqrt`](@ref), [`lbarker`](@ref), [`lmin1`](@ref), [`max1_logx`](@ref), [`max1`](@ref)
"""
lmax1(logx::T) where T <: Real = max(logx, zero(T))

"""
    max1_logx(logx::T) where T <: AbstractFloat

Compute max(x, 1) from log(x).

See also: [`lsqrt`](@ref), [`lbarker`](@ref), [`lmin1`](@ref), [`lmax1`](@ref), [`max1`](@ref)
"""
max1_logx(logx::T) where T <: AbstractFloat = logx < zero(T) ? one(T) : exp(logx)

"""
    max1(x::T) where T <: Real

Compute max(x, 1).

See also: [`sqrt`](@ref), [`barker`](@ref), [`min1`](@ref), [`lmax1`](@ref), [`max1_logx`](@ref)
"""
max1(x::T) where T <: Real = max(x, one(T))
