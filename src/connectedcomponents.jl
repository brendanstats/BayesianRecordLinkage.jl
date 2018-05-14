struct ConnectedComponents{G <: Integer}
    rowLabels::Array{G, 1}
    colLabels::Array{G, 1}
    rowperm::Array{G, 1}
    colperm::Array{G, 1}
    
    rowcounts::Array{G, 1}
    colcounts::Array{G, 1}
    cumrows::Array{G, 1}
    cumcols::Array{G, 1}
    
    nrow::G
    ncol::G
    ncomponents::G
end

function ConnectedComponents{G <: Integer}(rowLabels::Array{G, 1}, colLabels::Array{G, 1}, ncomponents::G = maximum(rowLabels), nrow::G = length(rowLabels), ncol::G = length(colLabels))

    rowperm = sortperm(rowLabels)
    colperm = sortperm(colLabels)    
    
    #off by one because zero is included as a label
    rowcounts = counts(rowLabels, 0:ncomponents)
    colcounts = counts(colLabels, 0:ncomponents)
    cumrows = cumsum(rowcounts)
    cumcols = cumsum(colcounts)

    return ConnectedComponents(rowLabels, colLabels, rowperm, colperm, rowcounts, colcounts, cumrows, cumcols, nrow, ncol, ncomponents)
end

function get_component(ii::Integer, cc::ConnectedComponents)
    if ii < zero(ii)
        warn("all labels are positive")
        rows = Int[]
        cols = Int[]
    elseif ii > cc.ncomponents
        warn("Label greater than number of connected components")
        rows = Int[]
        cols = Int[]
    elseif ii == zero(ii)
        rows = cc.rowLabels[cc.rowperm[1:cc.cumrows[1]]]
        cols = cc.colLabels[cc.colperm[1:cc.cumcols[1]]]
    else #+ 1 in range length to account for 0 labels
        rows = cc.rowperm[range(cc.cumrows[ii] + 1, cc.rowcounts[ii + 1])]
        cols = cc.colperm[range(cc.cumcols[ii] + 1, cc.colcounts[ii + 1])]
    end
    return rows, cols
end

function get_ranges(cc::ConnectedComponents)
    if issorted(cc.rowLabels) && issorted(cc.colLabels)
        out = Array{CartesianRange{CartesianIndex{2}}, 1}(cc.ncomponents)
        for ii in 1:cc.ncomponents
            rowrange = range(cc.cumrows[ii] + 1,cc.rowcounts[ii + 1])
            colrange = range(cc.cumcols[ii] + 1,cc.colcounts[ii + 1])
            out[ii] = CartesianRange((rowrange, colrange))
        end
    else
        error("input must be sorted by row and column cluster to summarize with ranges")
    end
    return out
end

#off by one because first component is labeled zero
function get_dimensions(ii::Integer, cc::ConnectedComponents)
    return cc.rowcounts[ii + 1], cc.colcounts[ii + 1]
end
