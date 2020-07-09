#########################################
#Setup
#########################################
using BayesianRecordLinkage, DelimitedFiles, StringDistances, Random

#########################################
#Comparison Vector Generation
#########################################

dataA, varA = readdlm("data/dataA.txt", '\t', String, header = true)
dataB, varB = readdlm("data/dataB.txt", '\t', String, header = true)

gnameInd = findfirst(x -> x == "gname", vec(varA))
fnameInd = findfirst(x -> x == "fname", vec(varA))
ageInd = findfirst(x -> x == "age", vec(varA))
occupInd = findfirst(x -> x == "occup", vec(varA))

nA = size(dataA, 1)
nB = size(dataB, 1)

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

function boolOrd(s1::String, s2::String)
    if s1 == "NA" || s2 == "NA"
        return Int8(0)
    elseif s1 == s2
        return Int8(1)
    else
        return Int8(2)
    end
end

ordArray = Array{Int8}(undef, nA, nB, 4)
for jj in 1:nB, ii in 1:nA
    ordArray[ii, jj, 1] = levOrd(dataA[ii, gnameInd], dataB[jj, gnameInd])
    ordArray[ii, jj, 2] = levOrd(dataA[ii, fnameInd], dataB[jj, fnameInd])
    ordArray[ii, jj, 3] = boolOrd(dataA[ii, ageInd], dataB[jj, ageInd])
    ordArray[ii, jj, 4] = boolOrd(dataA[ii, occupInd], dataB[jj, occupInd])
end

compsum = ComparisonSummary(ordArray)

##Set Penalized Likelihood Parameters
maxIter = 30
minincr = 0.1
penalty0 = 0.0
epsscale = 0.1

priorM = fill(1.01, sum(compsum.nlevels))
priorU = fill(1.01, sum(compsum.nlevels))

pM0 = prior_mode(priorM, compsum)
pU0 = prior_mode(compsum.counts, compsum)

##Demonstrate runtimes of penalized likelihood estimator with  different optimization approaches
#only one of these needs to be run for an actual analysis
@time penlikEst, penalties, iter = penalized_likelihood_search_hungarian(pM0, pU0, compsum, priorM, priorU, penalty0, minincr, maxIter = maxIter, cluster = false, verbose = false)
@time penlikEst, penalties, iter = penalized_likelihood_search_hungarian(pM0, pU0, compsum, priorM, priorU, penalty0, minincr, maxIter = maxIter, cluster = true, verbose = false)
@time penlikEst, penalties, iter = penalized_likelihood_search_auction(pM0, pU0, compsum, priorM, priorU, penalty0, minincr, maxIter = maxIter, cluster = false, update = false, epsiscale = 0.1, verbose = false)
@time penlikEst, penalties, iter = penalized_likelihood_search_auction(pM0, pU0, compsum, priorM, priorU, penalty0, minincr, maxIter = maxIter, cluster = false, update = true, epsiscale = 0.1, verbose = false)
@time penlikEst, penalties, iter = penalized_likelihood_search_auction(pM0, pU0, compsum, priorM, priorU, penalty0, minincr, maxIter = maxIter, cluster = true, update = false, epsiscale = 0.1, verbose = false)
@time penlikEst, penalties, iter = penalized_likelihood_search_auction(pM0, pU0, compsum, priorM, priorU, penalty0, minincr, maxIter = maxIter, cluster = true, update = true, epsiscale = 0.1, verbose = false)

##Set Post-hoc Blocking Parameters
npairscompThreshold = 2500
threshold0 = 0.0
thresholdInc = 0.01

maxW = maximum_weights_vector(penlikEst.pM, penlikEst.pU, compsum) #compute max weights
weightMat = penalized_weights_matrix(maxW, compsum, 0.0) #compute sparse matrix containing max weights
rowLabels, colLabels, maxLabel, clusterThresholds = iterative_bipartite_cluster2(weightMat, npairscompThreshold, threshold0, thresholdInc) #recursive clustering
cc = ConnectedComponents(rowLabels, colLabels, maxLabel) #construct connected components
phb = PosthocBlocks(cc, compsum) #construct post-hoc blocks

#Initialize using solution found with penalty = maximum posthoc blocking threshold
matchidx = findfirst(penalties .> maximum(clusterThresholds))
rows, cols = get_steplinks(matchidx, penlikEst)
C0 = LinkMatrix(BayesianRecordLinkage.tuple2links(rows, cols, compsum.nrow), compsum.ncol)


Random.seed!(29279)

##Run version where prior is computed throughout
lpriorC(nadd::Integer, C::LinkMatrix) = betabipartite_logratiopn(nadd, C, 1.0, 1.0)
runtime1 = @elapsed phbmcmc1, transCphb, Cphb = mh_gibbs_trace(25000, C0, compsum, phb, priorM, priorU, lpriorC, randomwalk1_locally_balanced_barker_update!)

##Run version where prior is pre-computed
priorComp = map(x -> betabipartite_logratiopn(1, x, compsum.nrow, compsum.ncol, 1.0, 1.0), 0:(compsum.nrow - 1))
runtime2 = @elapsed phbmcmc2, transCphb, Cphb = mh_gibbs_trace(25000, C0, compsum, phb, priorM, priorU, lpriorC, randomwalk1_locally_balanced_barker_update!)

println("Runtime of Restricted MCMC with Post-hoc for 25000 steps computing prior at each step: $runtime1")
println("Runtime of Restricted MCMC with Post-hoc for 25000 steps using precomputed prior: $runtime2")
