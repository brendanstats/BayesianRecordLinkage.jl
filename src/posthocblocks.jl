struct PosthocBlocks{G <: Integer}
    
    block2rows::Dict{G, Array{G, 1}}
    block2cols::Dict{G, Array{G, 1}}
    #row2blockidx::Array{G, 1}
    #col2blockidx::Array{G, 1}
    
    blocknrows::Array{G, 1}
    blockncols::Array{G, 1}
    blocksingleton::Array{Bool, 1}
    blocknnz::Array{Int, 1}

    nrow::G
    ncol::G
    nblock::G
    nnz::Int
end

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
    #row2blockidx = dict2keyidx(block2rows)
    #col2blockidx = dict2keyidx(block2cols)
    
    blocksingleton = [cc.rowcounts[kk + one(G)] == one(G) && cc.colcounts[kk + one(G)] == one(G)  for kk in one(G):cc.ncomponents]
    block2nnz = Int.(cc.rowcounts[2:end]) .* Int.(cc.colcounts[2:end])
    
    return PosthocBlocks(block2rows, block2cols, cc.rowcounts[2:end], cc.colcounts[2:end], blocksingleton, block2nnz, cc.nrow, cc.ncol, cc.ncomponents, length(compsum.obsidx))
end

function PosthocBlocks(cc::ConnectedComponents{G}, compsum::SparseComparisonSummary) where G <: Integer
    block2rows = label2dict(cc.rowLabels)
    block2cols = label2dict(cc.colLabels)
    #row2blockidx = dict2keyidx(block2rows)
    #col2blockidx = dict2keyidx(block2cols)
    
    blocksingleton = [cc.rowcounts[kk + one(G)] == one(G) && cc.colcounts[kk + one(G)] == one(G)  for kk in one(G):cc.ncomponents]
    block2nnz = zeros(Int, cc.ncomponents)
    for kk in one(G):cc.ncomponents
        for col in block2cols[kk]
            block2nnz[kk] += count(rowvals(compsum.obsidx)[nzrange(compsum.obsidx, col)] .== kk)
        end
    end

    return PosthocBlocks(block2rows, block2cols, cc.rowcounts[2:end], cc.colcounts[2:end], blocksingleton, block2nnz, cc.nrow, cc.ncol, cc.ncomponents, length(compsum.obsidx))
end

function SparseComparisonSummary(compsum::ComparisonSummary{G, T}, phb::PosthocBlocks{A}) where {G <: Integer, T <: Integer, A <: Integer}
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
