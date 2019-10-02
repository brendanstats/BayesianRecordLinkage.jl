"""
    struct PosthocBlocks{G <: Integer}

Store blocking structure in a manner that makes retrival of the entries within a block efficient.

If constructor is called without a `ConnectedComponents` object then only a single block, containing
all rows and all columns is created.  Otherwise blocks correspond to components within the
`ConnectedComponents` object. Primaryilu 


# Fields

* `block2rows::Dict{G, Array{G, 1}}`: Mapping from block to set of rows contained in the block.
* `block2cols::Dict{G, Array{G, 1}}`: Mapping from block to set of columns contained in the block.
* `blocknrows::Array{G, 1}`: Number of rows contained in the block, blocknrows[kk] == length(block2rows[kk]).
* `blockncols::Array{G, 1}`: Number of columns contained in the block, blockncols[kk] == length(block2cols[kk])
* `blocksingleton::Array{Bool, 1}`: blocksingleton[kk] == true if blocknrows[kk] == 1 and block2cols[kk] == 1.
* `blocknnz::Array{Int, 1}`: Number of non-zero entries contained in the block.  This will be blocknrows[kk] * blockncols[kk] unless the constructure is run with a `SparseComparisonSummary` in which case the number of non-zero entries in the sparse matrix region corresponding to the block will be counted.
* `nrow::G`: Number of rows in array containing blocks, all entries in block2rows should be <= nrow.
* `ncol::G`:  Number of columns in array containing blocks, all entries in block2cols should be <= ncol.
* `nblock::G`: Number of blocks, nblock == maximum(keys(block2rows)) == maximum(keys(block2cols))
* `nnz::Int`: Total number of entires, equal to sum(blocknnz).

# Constructors

    PosthocBlocks(compsum::Union{ComparisonSummary, SparseComparisonSummary})
    PosthocBlocks(cc::ConnectedComponents{G}, compsum::ComparisonSummary) where G <: Integer
    PosthocBlocks(cc::ConnectedComponents{G}, compsum::SparseComparisonSummary) where G <: Integer

# Arguments

* `compsum::Union{ComparisonSummary, SparseComparisonSummary}`: Summary of comparisons between record pairs.
* `cc::ConnectedComponents{G}`: Results of graph clustering algorithm denoting blocks.

See also: [`ConnectedComponents`](@ref), [`mh_gibbs_trace`](@ref), [`mh_gibbs_count`](@ref)
"""
struct PosthocBlocks{G <: Integer}
    
    block2rows::Dict{G, Array{G, 1}}
    block2cols::Dict{G, Array{G, 1}}
    
    blocknrows::Array{G, 1}
    blockncols::Array{G, 1}
    blocksingleton::Array{Bool, 1}
    blocknnz::Array{Int, 1}

    nrow::G
    ncol::G
    nblock::G
    nnz::Int
end

"""
    label2dict(x::Array{G}) where G <: Integer

Take an integer array and constructed a dictionary mapping from values in `x` to the set of indicies.

See also: [`dict2keyidx`](@ref)
"""
function label2dict(x::Array{G}) where G <: Integer
    labelMap = Dict{G, Array{G, 1}}()
    for ii in one(G):G(length(x))
        if haskey(labelMap, x[ii])
            push!(labelMap[x[ii]], ii)
        else
            labelMap[x[ii]] = G[ii]
        end
    end
    return labelMap
end

"""
    dict2keyidx(d::Dict{G, Array{G, 1}}, n::G) where G <: Integer

Inverse of `label2dict` taking a dictionary and returning an array where values are determined by key which maps to index.

See also: [`label2dict`](@ref)
"""
function dict2keyidx(d::Dict{G, Array{G, 1}}, n::G) where G <: Integer
    out = zeros(G, n)
    for kk in  keys(d)
        for ii in 1:length(d[kk])
            out[d[kk][ii]] = ii
        end
    end
    return out
end

function PosthocBlocks(compsum::Union{ComparisonSummary, SparseComparisonSummary})
    PosthocBlocks(Dict(1 => collect(1:compsum.nrow)), Dict(1 => collect(1:compsum.ncol)), [compsum.nrow], [compsum.ncol], [compsum.nrow == 1 && compsum.ncol == 1], [compsum.npairs], compsum.nrow, compsum.ncol, 1, compsum.npairs)
end

function PosthocBlocks(cc::ConnectedComponents{G}, compsum::ComparisonSummary) where G <: Integer
    block2rows = label2dict(cc.rowLabels)
    block2cols = label2dict(cc.colLabels)
    
    blocksingleton = [cc.rowcounts[kk + one(G)] == one(G) && cc.colcounts[kk + one(G)] == one(G)  for kk in one(G):cc.ncomponents]
    block2nnz = Int.(cc.rowcounts[2:end]) .* Int.(cc.colcounts[2:end])
    
    return PosthocBlocks(block2rows, block2cols, cc.rowcounts[2:end], cc.colcounts[2:end], blocksingleton, block2nnz, cc.nrow, cc.ncol, cc.ncomponents, length(compsum.obsidx))
end

function PosthocBlocks(cc::ConnectedComponents{G}, compsum::SparseComparisonSummary) where G <: Integer
    block2rows = label2dict(cc.rowLabels)
    block2cols = label2dict(cc.colLabels)
    
    blocksingleton = [cc.rowcounts[kk + one(G)] == one(G) && cc.colcounts[kk + one(G)] == one(G)  for kk in one(G):cc.ncomponents]
    block2nnz = zeros(Int, cc.ncomponents)
    for kk in one(G):cc.ncomponents
        for col in block2cols[kk]
            block2nnz[kk] += count(cc.rowLabels[rowvals(compsum.obsidx)[nzrange(compsum.obsidx, col)]] .== kk)
        end
    end

    return PosthocBlocks(block2rows, block2cols, cc.rowcounts[2:end], cc.colcounts[2:end], blocksingleton, block2nnz, cc.nrow, cc.ncol, cc.ncomponents, sum(block2nnz))
end
