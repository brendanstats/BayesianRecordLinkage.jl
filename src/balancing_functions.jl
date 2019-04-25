####################
#identity (globally)
####################

#logs
"""
    lidentity_balance(loglikMargin::AbstractFloat)
    lidentity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1})
    lidentity_balance(loglikMargin::AbstractFloat, nadd::Integer, C::LinkMatrix, logpdfC::Function)
    lidentity_balance(loglikMargin::AbstractFloat, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
    lidentity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
### Arguments

* `var` : brief description

### Details

### Value

### Examples

```julia

```
"""
lidentity_balance(loglikMargin::AbstractFloat) = loglikMargin
lidentity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}) = dot(delta, logDiff)

lidentity_balance(loglikMargin::AbstractFloat, nadd::Integer, C::LinkMatrix, logpdfC::Function) =
    loglikMargin + logpdfC(nadd, C)

function lidentity_balance(loglikMargin::AbstractFloat, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
    return ratioPrior ? (loglikMargin + logpdfC(propC, C)) : (loglikMargin + logpdfC(propC) - logpdf(C))
end

function lidentity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, nadd::Integer, C::LinkMatrix, logpdfC::Function)
    return dot(delta, logDiff) + logpdfC(nadd, C)
end

function lidentity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool)
    return ratioPrior ? (dot(delta, logDiff) + logpdfC(propC, C)) : (dot(delta, logDiff) + logpdfC(propC) - logpdf(C))
end

#base

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
identity_balance(loglikMargin::AbstractFloat) = exp(loglikMargin)
identity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}) = exp(dot(delta, logDiff))
identity_balance(loglikMargin::AbstractFloat, nadd::Integer, C::LinkMatrix, logpdfC::Function) =
    exp(lidentity_balance(loglikMargin, nadd, C, logpdfC))
identity_balance(loglikMargin::AbstractFloat, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) =
    exp(lidentity_balance(loglikMargin, propC, C, logpdfC, ratioPrior))
identity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, nadd::Integer, C::LinkMatrix, logpdfC::Function) = exp(lidentity_balance(delta, logDiff, nadd, C, logpdfC))
identity_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) = exp(lidentity_balance(delta, logDiff, propC, C, logpdfC, ratioPrior))

####################
#sqrt
####################

#logs
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
lsqrt_balance(loglikMargin::AbstractFloat) = 0.5 * loglikMargin
lsqrt_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}) = 0.5 * dot(delta, logDiff)
lsqrt_balance(loglikMargin::AbstractFloat, nadd::Integer, C::LinkMatrix, logpdfC::Function) =
    0.5 * lidentity_balance(loglikMargin, nadd, C, logpdfC)
lsqrt_balance(loglikMargin::AbstractFloat, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) =
    0.5 * lidentity_balance(loglikMargin, propC, C, logpdfC, ratioPrior)
lsqrt_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, nadd::Integer, C::LinkMatrix, logpdfC::Function) = 0.5 * lidentity_balance(delta, logDiff, nadd, C, logpdfC)
lsqrt_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) = 0.5 * lidentity_balance(delta, logDiff, propC, C, logpdfC, ratioPrior)

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
sqrt_balance(loglikMargin::AbstractFloat) = exp(0.5 * loglikMargin)
sqrt_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}) = exp(0.5 * dot(delta, logDiff))
sqrt_balance(loglikMargin::AbstractFloat, nadd::Integer, C::LinkMatrix, logpdfC::Function) =
    exp(lsqrt_balance(loglikMargin, nadd, C, logpdfC))
sqrt_balance(loglikMargin::AbstractFloat, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) =
    exp(lsqrt_balance(loglikMargin, propC, C, logpdfC, ratioPrior))
sqrt_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, nadd::Integer, C::LinkMatrix, logpdfC::Function) = exp(lsqrt_balance(delta, logDiff, nadd, C, logpdfC))
sqrt_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) = exp(lsqrt_balance(delta, logDiff, propC, C, logpdfC, ratioPrior))

####################
#barker
####################

lbarker(x::AbstractFloat) = x - log1pexp(x)

#logs

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
lbarker_balance(loglikMargin::AbstractFloat) = lbarker(loglikMargin)
lbarker_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}) = lbarker(dot(delta, logDiff))
lbarker_balance(loglikMargin::AbstractFloat, nadd::Integer, C::LinkMatrix, logpdfC::Function) =
    lbarker(lidentity_balance(loglikMargin, nadd, C, logpdfC))
lbarker_balance(loglikMargin::AbstractFloat, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) =
    lbarker(lidentity_balance(loglikMargin, propC, C, logpdfC, ratioPrior))
lbarker_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, nadd::Integer, C::LinkMatrix, logpdfC::Function) = lbarker(lidentity_balance(delta, logDiff, nadd, C, logpdfC))
lbarker_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) = lbarker(lidentity_balance(delta, logDiff, propC, C, logpdfC, ratioPrior))

#logistic(log(t)) = t / (t + 1)

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
barker_balance(loglikMargin::AbstractFloat) = logistic(loglikMargin)
barker_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}) = logistic(dot(delta, logDiff))
barker_balance(loglikMargin::AbstractFloat, nadd::Integer, C::LinkMatrix, logpdfC::Function) =
    logistic(lidentity_balance(loglikMargin, nadd, C, logpdfC))
barker_balance(loglikMargin::AbstractFloat, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) =
    logistic(lidentity_balance(loglikMargin, propC, C, logpdfC, ratioPrior))
barker_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, nadd::Integer, C::LinkMatrix, logpdfC::Function) = logistic(lidentity_balance(delta, logDiff, nadd, C, logpdfC))
barker_balance(delta::Array{<:Integer, 1}, logDiff::Array{<:AbstractFloat, 1}, propC::LinkMatrix, C::LinkMatrix, logpdfC::Function, ratioPrior::Bool) = logistic(lidentity_balance(delta, logDiff, propC, C, logpdfC, ratioPrior))
