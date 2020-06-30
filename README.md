# BayesianRecordLinkage
Collection of methods for performing Bayesian inference for large-scale record linkage problems as described in [Scaling Bayesian Probabilistic Record Linkage with Post-Hoc Blocking: An Application to the California Great Registers](https://arxiv.org/abs/1905.05337).  

## Install Julia

Install Julia for Mac, Windows, or Linux at https://julialang.org/downloads/

A tutorial on Julia can be found here: https://www.sas.upenn.edu/~jesusfv/Chapter_HPC_8_Julia.pdf


## Installation
```julia
] # = allows on to install packages 
pkg> add https://github.com/brendanstats/AssignmentSolver.jl
pkg> add https://github.com/brendanstats/BayesianRecordLinkage.jl
```

### Dependencies
In addition to the `AssignmentSolver` module the following packages are loaded: `Distributions`, `DataStructures`, `StatsBase`, `StatsFuns`, `SparseArrays`, `SpecialFunctions`, `Dates` `StringDistances`.  These should be installed automatically by the package manager but can be installed with the following line if necessary.

```julia
pkg> add Distributions DataStructures StatsBase StatsFuns SparseArrays SpecialFunctions Dates StringDistances
```

# Methods
The package implements a variety of methds described in [McVeigh, Spahn, & Murray (2019)](https://arxiv.org/abs/1905.05337) that allow probabilistic Bayesian record linkage models to be estimated for datasets involving hundreds of thousands or millions of records.  Throughout a conditional independece model for record linkage employing categorical comparison vectors is assumed.  This model is identical to the one outlined in Sadinle (2017).

## Penalized-likelihood Estimator
The `penalized_likelihood_search_hungarian` and `penalized_likelihood_search_auction` functions allow for a sequence of model estimates with the number of links subject to increasingly large penalty terms. The information from this sequence can then be used in construting a set of post-hoc blocks.  The `penalized_likelihood_search_hungarian` function relies on a Hungarian algorithm to solve assignment problems internally while `penalized_likelihood_search_auction` relies on an auction algorith.  In most cases the auction algorithm will be dramatically more computationally efficient and it is therefore recommended that the `penalized_likelihood_search_auction` function be used.

## Post-hoc Blocking with Restricted Markov chain Monte Carlo algorithm
The `mh_gibbs_trace` and `mh_gibbs_count` functions implement an MCMC sampler for estimating a posterior distribution over the linkage structure and the model parameters.  `mh_gibbs_trace` retirns the full trace of the linkage structure, storing link persistence, when specifc record pairs are added and deleted, whereas `mh_gibbs_count` stores only the number of MCMC iterations in which each record pair is presemt.  Both functions save a full trace of the other model parameters.  Additionally, both return a set of three objects.  The first returned object (of type `ParameterChain`) contains the trace (or counts) of the link structure and the trace of the other model parameters.  The second returned object contains the frequencies with which updates were performed for each post-hoc block. The third returned object contains the link structure (of type `LinkMatrix`) after the final step of the sampler.

### MCMC Moves
Within the MCMC sampler the model parameters are updated, conditional on the linkage structure via a gibbs update.  For singleton post-hoc blocks, those containing only a single record pair, a gibbs update to the linkage structure within the block is also possible.  For larger post-hoc blocks locally balanced moves [Zanella 2019](https://www.tandfonline.com/doi/full/10.1080/01621459.2019.1585255) are used.

# Prior Distributions over Linkage Structure
Currently functions are provided to evaluate two prior distributions over the linkage structure, the beta prior for bipartite matchings (Sadinle 2017) and a prior which applies an exponential penalty (Green and Mardia 2006).

# Example
Here we provide an example 
```julia
########################################
#Setup
########################################
using BayesianRecordLinkage, DelimitedFiles, StringDistances, Random

########################################
# Comparison Vector Generation
########################################
## Load data
dataA, varA = readdlm("data/dataA.txt", '\t', String, header = true)
dataB, varB = readdlm("data/dataB.txt", '\t', String, header = true)

nA = size(dataA, 1)
nB = size(dataB, 1)

## Find fields to compare
gnameInd = findfirst(x -> x == "gname", vec(varA))
fnameInd = findfirst(x -> x == "fname", vec(varA))
ageInd = findfirst(x -> x == "age", vec(varA))
occupInd = findfirst(x -> x == "occup", vec(varA))

## Define function to generate an ordinal comparison between two strings using a Levenshtein sting distance
function levOrd(s1::String, s2::String)
    levsim = compare(Levenshtein(), s1, s2)
    if levsim == 1.0
        return Int8(1)
    elseif levsim >= 0.75
        return Int8(2)
    elseif levsim >= 0.5
        return Int8(3)
    else
        return Int8(4)
    end
end

## Define function to generate an ordinal comparison between two strings using exact matching
function boolOrd(s1::String, s2::String)
    if s1 == "NA" || s2 == "NA"
        return Int8(0)
    elseif s1 == s2
        return Int8(1)
    else
        return Int8(2)
    end
end

## Generate ordinal comparisons
ordArray = Array{Int8}(undef, nA, nB, 4)
for jj in 1:nB, ii in 1:nA
    ordArray[ii, jj, 1] = levOrd(dataA[ii, gnameInd], dataB[jj, gnameInd])
    ordArray[ii, jj, 2] = levOrd(dataA[ii, fnameInd], dataB[jj, fnameInd])
    ordArray[ii, jj, 3] = boolOrd(dataA[ii, ageInd], dataB[jj, ageInd])
    ordArray[ii, jj, 4] = boolOrd(dataA[ii, occupInd], dataB[jj, occupInd])
end

## Load into a ComparsionSummary object
compsum = ComparisonSummary(ordArray)

########################################
#Penalized-Likelihood Estimation
########################################

## Set Penalized Likelihood Parameters
maxIter = 30 #Maximum number of interations
minincr = 0.1 #Increment of penalty
penalty0 = 0.0 #Starting penalty value
epsscale = 0.1 #Epsilon scaling value used within auction algorithm

## Define flat prios 
priorM = fill(1.0, sum(compsum.nlevels))
priorU = fill(1.0, sum(compsum.nlevels))

## Compute prior modes for each model parameter, assumes a Dirichlet prior.
pM0 = prior_mode(priorM, compsum)
pU0 = prior_mode(compsum.counts, compsum)

## Run penalized likelihood estimator
penlikseq, penalties, iter = penalized_likelihood_search_auction(pM0, pU0, compsum, priorM, priorU, penalty0, minincr, maxIter = maxIter, cluster = false, update = false, epsiscale = 0.1, verbose = false)

########################################
#Post-hoc Blocking
########################################

## Set Post-hoc Blocking Parameters
npairscompThreshold = 2500 #maximum number of record pairs allowed in a post-hoc block (50^2)
threshold0 = 0.0 #Initial weight threshold for clustering
thresholdInc = 0.01 #Threshold increment for breaking larger blocks

## Find post-hoc blocks using maximum weights across penalized likelihood sequence as edge weights
maxW = maximum_weights_vector(penlikseq.pM, penlikseq.pU, compsum)
weightMat = penalized_weights_matrix(maxW, compsum, 0.0)
rowLabels, colLabels, maxLabel, clusterThresholds = iterative_bipartite_cluster2(weightMat, npairscompThreshold, threshold0, thresholdInc)
cc = ConnectedComponents(rowLabels, colLabels, maxLabel)
phblocks = PosthocBlocks(cc, compsum)

########################################
#Restricted MCMC
########################################

## Initialize using solution found with penalty = maximum posthoc blocking threshold
matchidx = findfirst(penalties .> maximum(clusterThresholds))
rows, cols = get_steplinks(matchidx, penlikseq)
C0 = LinkMatrix(BayesianRecordLinkage.tuple2links(rows, cols, compsum.nrow), compsum.ncol)

## Set seed
Random.seed!(29279)

## Set flat prior over the share of record pairs linked
lpriorC(nadd::Integer, C::LinkMatrix) = betabipartite_logratiopn(nadd, C, 1.0, 1.0)

## Run restricted MCMC sampler
postchain, transC, C = mh_gibbs_trace(25000, C0, compsum, phblocks, priorM, priorU, lpriorC, randomwalk1_locally_balanced_barker_update!)

## Compute frequency with which each record pair is linked and return counts for all pairs with nonzero counts
cts = get_linkcounts(postchain)
cts = cts[cts[:, 3] .> 12500, :] #reduce to just record pairs linked more than half the time (the Bayes estimator)
Cbayes =  LinkMatrix(nA, nB, cts[:, 1], cts[:, 2])
ntp = count(Cbayes.row2col[1:300] .== 1:300) #Example constructed so that rows 1:300 in A match rows 1:300 in B
nfp = Cbayes.nlink - ntp

println("Precision: $(ntp / Cbayes.nlink)")
println("Recall: $(ntp / 300)")
```

## Blocking or Indexing
The `SparseComparisonSummary` type is avaiable for problems where comparison vectors are not computed for all possible record pairs.  An example of this is shown below, note that the example was constructed for simplicty and more efficient methods of blockng or indexing should be used in practice.  The `SparseComparisonSummary` object can then be used in place of a standard `ComparisonSummary` object.
```julia
## Generate ordinal comparisons
indsA = Int32[]
indsB = Int32[]
for jj in 1:nB, ii in 1:nA
    gnameComp = compare(Levenshtein(), dataA[ii, gnameInd], dataB[jj, gnameInd])
    fnameComp = compare(Levenshtein(), dataA[ii, fnameInd], dataB[jj, fnameInd])
    if (gnameComp > 0.6) || (fnameComp > 0.6)
        push!(indsA, ii)
        push!(indsB, jj)
    end
end

## Generate ordinal comparisons
ordArray = Array{Int8}(undef, length(indsA), 4)
for ii in 1:length(indsA)
    ordArray[ii, 1] = levOrd(dataA[indsA[ii], gnameInd], dataB[indsB[ii], gnameInd])
    ordArray[ii, 2] = levOrd(dataA[indsA[ii], fnameInd], dataB[indsB[ii], fnameInd])
    ordArray[ii, 3] = boolOrd(dataA[indsA[ii], ageInd], dataB[indsB[ii], ageInd])
    ordArray[ii, 4] = boolOrd(dataA[indsA[ii], occupInd], dataB[indsB[ii], occupInd])
end

spcompsum = SparseComparisonSummary(indsA, indsB, ordArray, size(dataA, 1), size(dataB, 1), [4, 4, 2, 2])
```

# Custom Types

## ComparisonSummary
* `obsidx::Array{T, 2}`: mapping from record pair to comparison vector, indicates which column of obsvecs contains the original record pair
* `obsvecs::Array{G, 2}`: comparison vectors observed in data, each column corresponds to an observed comparison vector. Zero entries indicate a missing value for the comparison-feature pair.
* `obsvecct::Array{Int64, 1}`: number of observations of each comparison vector, length = size(obsvecs, 2)
* `counts::Array{Int64, 1}`: vector storing counts of all comparisons for all levels of all fields, length = sum(nlevels)
* `obsct::Array{Int64, 1}`: number of observations for each comparison field, length(obsct) = ncomp)
* `misct::Array{Int64, 1}`: number of comparisons missing for each comparison, length(obsct) = ncomp, obsct[ii] + misct[ii] = npairs)
* `nlevels::Array{Int64, 1}`: number of levels allowed for each comparison
* `cmap::Array{Int64, 1}`: which comparison does counts[ii] correspond to
* `levelmap::Array{Int64, 1}`: which level does counts[ii] correspond to
* `cadj::Array{Int64, 1}`: cadj[ii] + jj yields the index in counts for the jjth level of comparison ii, length(cadj) = ncomp, use range(cadj[ii] + 1, nlevels[ii]) to get indexes for counts of comparison ii
* `nrow::Int64`: number of records in first data base, first dimension of input Array
* `ncol::Int64`: number of records in second data base, second dimension of input Array
* `npairs::Int64`: number of records pairs = nrow * ncol
* `ncomp::Int64`: number of comparisons, computed as third dimension of input Array 

## SparseComparisonSummary

* `obsidx::SparseMatrixCSC{Tv, Ti}`: mapping from record pair to comparison vector, indicates which column of obsvecs contains the original record pair
* `obsvecs::Array{G, 2}`: comparison vectors observed in data, each column corresponds to an observed comparison vector.  Zero entries indicate a missing value for the comparison-feature pair.
* `obsvecct::Array{Int64, 1}`: number of observations of each comparison vector, length = size(obsvecs, 2)
* `counts::Array{Int64, 1}`: vector storing counts of all comparisons for all levels of all fields, length = sum(nlevels)
* `obsct::Array{Int64, 1}`: number of observations for each comparison field, length(obsct) = ncomp)
* `misct::Array{Int64, 1}`: number of comparisons missing for each comparison, length(obsct) = ncomp, obsct[ii] + misct[ii] = npairs)
* `nlevels::Array{Int64, 1}`: number of levels allowed for each comparison
* `cmap::Array{Int64, 1}`: which comparison does counts[ii] correspond to
* `levelmap::Array{Int64, 1}`: which level does counts[ii] correspond to
* `cadj::Array{Int64, 1}`: cadj[ii] + jj yields the index in counts for the jjth level of comparison ii, length(cadj) = ncomp, use range(cadj[ii] + 1, nlevels[ii]) to get indexes for counts of comparison ii
* `nrow::Int64`: number of records in first data base, first dimension of input Array
* `ncol::Int64`: number of records in second data base, second dimension of input Array
* `npairs::Int64`: number of records pairs = nrow * ncol
* `ncomp::Int64`: number of comparisons, computed as third dimension of input Array

## LinkMatrix
The `LinkMatrix` type stores the current state of the linkage structure and has the following fields:
* `row2col::Array{G, 1}`: Mapping from row to column.  row2ccol[ii] = jj if row ii is linked to column jj.  If row ii is unlinked then row2col[ii] == 0.
* `col2row::Array{G, 1}`: Mapping from column to row.  col2row[jj] = ii if column jj is linked to row ii.  If column jj is unlinked then col2row[jj] == 0.       
* `nlink::G`: Number of record pairs counted as links.
* `nrow::G`: Number of rows equal, nrow == length(row2col).
* `ncol::G`: Number of columns, ncol == length(col2row) 

## ConnectedComponents

* `rowLabels::Array{G, 1}`: Component assignment of row, zero indicates no edges from row.
* `colLabels::Array{G, 1}`: Component assignment of col, zero indicates no edges from column.
* `rowperm::Array{G, 1}`: Permutation which will reorder rows so that labels are ascending.
* `colperm::Array{G, 1}`: Permutation which will reorder columns so that labels are ascending.
* `rowcounts::Array{G, 1}`: Number of rows contained in each component, because counts forunassigned (label == 0) rows are included rowcounts[kk + 1] contains the count for component kk.
* `colcounts::Array{G, 1}`: Number of columns contained in each component, because counts forunassigned (label == 0) columns are included colcounts[kk + 1] contains the count for component kk.
* `cumrows::Array{G, 1}`: Cumulative sum of `rowcounts`.
* `cumcols::Array{G, 1}`: Cumulative sum of `colcounts`.
* `nrow::G`: Number of rows (nodes from first group).
* `ncol::G`: Number of columns (nodes from second group).
* `ncomponents::G`: Number of components.   

## PosthocBlocks
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

## ParameterChain
The `ParameterChain` type will store a trace of both the linkage structure and the model parameters when the `linktrace` field is set to `true`.  When the `linktrace` field is set to false then only counts of the frequency with which a given record pair was linked will be stores.  The type contains the following fields:

* `C::Array{G, 2}`: Either row and column indices paired with link counts or indicies paired with steps / iterations in which they appeared.
* `nlinks::Union{Array{G, 1}, Array{G, 2}}`: Total number of links at each step / iteration.
* `pM::Array{T, 2}`: M parameters at each iteration, size(pM, 1) == nsteps
* `pU::Array{T, 2}`: U parameters at each iteration, size(pU, 1) == nsteps
* `nsteps::Int`: Total number of steps / iterations
* `linktrace::Bool`: Indicator if `C` contains link counts or a trace of the entire link structure.

# References
1. Green, P. J. and Mardia, K. V. 2006,  Bayesian alignment using hierarchical models, with applications in protein bioinformatics, Biometrika, 93(2):235–254.
2. McVeigh, B.S., Spahn, B.T. and Murray, J.S., 2019, Scaling Bayesian Probabilistic Record Linkage with Post-Hoc Blocking: An Application to the California Great Registers, arXiv preprint arXiv:1905.05337
3. Mauricio Sadinle 2017, Bayesian Estimation of Bipartite Matchings for Record Linkage, Journal of the American Statistical Association, 112:518, 600-612, DOI: 10.1080/01621459.2016.1148612
4. Giacomo Zanella 2019, Informed Proposals for Local MCMC in Discrete Spaces, Journal of the American Statistical Association, DOI: 10.1080/01621459.2019.1585255
