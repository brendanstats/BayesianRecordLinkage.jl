using One2OneRecordLinkage, Base.Test
using RCall
R"library(clue)"
#include("test_comparisonsummary.jl")
#include("test_em_functions.jl")
#include("test_lsap_solver.jl")
#include("test_clustering_functions.jl")
#include("test_map_functions.jl")

########################################
#Test comparisonsummary.jl functions
########################################
##ComparisonSummary(bool array)
n, m = 34, 45
comparisons = bitrand(n,m,3)
compsum = ComparisonSummary(comparisons)
for jj in 1:m, ii in 1:n
    @test compsum.obsvecs[:, compsum.obsidx[ii,jj]] == comparisons[ii, jj, :]
end
@test size(compsum.obsvecs, 1) == 3
@test length(compsum.obsvecct)

##ComparisonSummary(integer array)

##ComparisonSummary(integer array) w/ missing data coded as zeros

########################################
#Test em_functions.jl functions
########################################
#E_step(pM, pU, p, ComparisonSummary)
#M_step(gM, gU, ComparisonSummary, pseudoM, pseudoU)
#estimate_EM(pM0, pU0, p0, ComparisonSummary, maxIter, pseudoM, pseudoU, tol)

########################################
#Test clustering_functions.jl functions
########################################

@testset "Bipartite Clustering" begin

    ##Unit tests for binary clustering
    @testset "Unit Tests" begin
        @test bipartite_cluster([false false; false false]) == ([0, 0], [0, 0], 0)
        @test bipartite_cluster([false true; false false]) == ([1, 0], [0, 1], 1)
        @test bipartite_cluster([true true; false false]) == ([1, 0], [1, 1], 1)
        @test bipartite_cluster([true true; true false]) == ([1, 1], [1, 1], 1)
        @test bipartite_cluster([false true; true false]) == ([2, 1], [1, 2], 2)
        @test bipartite_cluster([true false; false true]) == ([1, 2], [1, 2], 2)
    end

    ##Random tests for Float - Threshold clustering
    @testset "Random Tests" begin
        for ii in [10, 50, 100, 250, 500], jj in [10, 50, 100, 250, 500], kk in 0.0:0.1:1.0
            costs = rand(ii, jj)
            @test bipartite_cluster(costs, kk) == bipartite_cluster(costs .> kk)
            @test bipartite_cluster(costs, kk) == bipartite_cluster(sparse(costs .> kk))
        end
    end

end

########################################
#Test prior_functions.jl
########################################

########################################
#Test balancing_functions.jl
########################################

########################################
#Test move_functions.jl
########################################

#check weights                                                                                                                                                                  
@time w1 = One2OneRecordLinkage.log_move_weights(C, compsum, loglikMargin, logpdfC, lidentity_balance)

@time w2 = One2OneRecordLinkage.move_weights(C, compsum, countDeltas, logDiff, logratioC, true, identity_balance)

maximum(abs.(exp.(w1) - w2))

@time w1 = One2OneRecordLinkage.log_move_weights(C, compsum, loglikMargin, logpdfC, lsqrt_balance)

@time w2 = One2OneRecordLinkage.move_weights(C, compsum, countDeltas, logDiff, logratioC, true, sqrt_balance)

maximum(abs.(exp.(w1) - w2))

@time w1 = One2OneRecordLinkage.log_move_weights(C, compsum, loglikMargin, logpdfC, lbarker_balance)

@time w2 = One2OneRecordLinkage.move_weights(C, compsum, countDeltas, logDiff, logratioC, true, barker_balance)

maximum(abs.(exp.(w1) - w2))


########################################
#Test map_functions.jl
########################################

#count_matches(matchRows, matchColumns, comparisonSummary) -> matchCounts, obsCounts
#max_MU(matchRows, matchColumns, comparisonSummary, pseudoM, pseudoU) -> pM, pU
#compute_weights(pM, pU, comparisonSummary) -> weightArray
#compute_penalized_weights(pM, pU, comparisonSummary, penalty) -> weightArray
#compute_costs(pM, pU, comparisonSummary, penalty) -> costArray, maxcost
#max_C(pM, pU, comparisonSummary, penalty) -> matchRows, matchColumns
#max_C_complete(pM, pU, comparisonSummary, penalty) -> matchRows, matchColumns, keep, costs
#map_solver(pM0, pU0, comparisonSummary, [priorM], [priorU], penalty; maxIter) -> matchRows, matchColumns, pM, pU, iterations
#map_solver_iter(pM0, pU0, comparisonSummary, [priorM], [priorU], penaltyRng; maxIter) -> matchRows, matchColumns, pM, pU, iterations

using BayesianRecordLinkage, DelimitedFiles

##Priors                                                                                                                                                        
priorM = repeat([1.9, 1.1], outer = 3)
priorU = repeat([1.1, 1.9], outer = 3)
lpriorC(nadd::Integer, C::LinkMatrix) = betabipartite_logratiopn(nadd, C, 1.0, 1.0)

##Initial values                                                                                                                                                
pM0 = repeat([0.9, 0.1], outer = 3)
pU0 = repeat([0.1, 0.9], outer = 3)
p0 = 30.0 / 1530.0

########################################
#Load data and generate comparison vectors
########################################

A = readdlm("data/exampleA.dat", Int64, header = true)[1][:, 2:end]
B = readdlm("data/exampleB.dat", Int64, header = true)[1][:, 2:end]

permA = sortperm(A[:, 2])
permB = sortperm(B[:, 2])

A = A[permA, :]
B = B[permB, :]

data = [A[ii, kk] == B[jj, kk] for ii in 1:size(A, 1), jj in 1:size(B, 1), kk in 1:size(A, 2)]
compsum = ComparisonSummary(data)

priorM = repeat([1.9, 1.1], outer = 3)
priorU = repeat([1.1, 1.9], outer = 3)
pM0 = repeat([0.9, 0.1], outer = 3)
pU0 = repeat([0.1, 0.9], outer = 3)
penalty0 = 0.0
tol = 0
digt = 5
minmargin = 0.0
maxIter = 100
cluster = false
update = false

wpenalized = penalized_weights_vector(pM0, pU0, compsum, penalty0)

@time r2c, pM, pU, iter = map_solver(pM0, pU0, compsum, priorM, priorU, 0.0)
@time r2c, pM, pU, iter = map_solver(pM, pU, compsum, priorM, priorU, 0.24)
@time r2c, pM, pU, iter = map_solver(pM, pU, compsum, priorM, priorU, 2.552)
pM
count(.!iszero.(r2c))

@time astate, pM, pU, iter = penalized_likelihood_hungarian(pM0, pU0, compsum, priorM, priorU, 0.0, cluster = false)
@time astate, pM, pU, iter = penalized_likelihood_hungarian(pM, pU, compsum, priorM, priorU, 0.24, cluster = false)
@time astate, pM, pU, iter = penalized_likelihood_hungarian(pM, pU, compsum, priorM, priorU, 2.552, cluster = false)
pM
astate.nassigned

@time astate, pM, pU, iter = penalized_likelihood_hungarian(pM0, pU0, compsum, priorM, priorU, 0.0, cluster = true)
@time astate, pM, pU, iter = penalized_likelihood_hungarian(pM, pU, compsum, priorM, priorU, 0.24, cluster = true)
@time astate, pM, pU, iter = penalized_likelihood_hungarian(pM, pU, compsum, priorM, priorU, 2.552, cluster = true)
pM
astate.nassigned

@time astate, pM, pU, iter = penalized_likelihood_cluster_hungarian(pM0, pU0, compsum, priorM, priorU, 0.0)
@time astate, pM, pU, iter = penalized_likelihood_cluster_hungarian(pM, pU, compsum, priorM, priorU, 0.24)
@time astate, pM, pU, iter = penalized_likelihood_cluster_hungarian(pM, pU, compsum, priorM, priorU, 2.552)
pM
astate.nassigned

@time astate, pM, pU, iter = penalized_likelihood_auction(pM0, pU0, compsum, priorM, priorU, 0.0, cluster = false, update = false)
@time astate, pM, pU, iter = penalized_likelihood_auction(pM, pU, compsum, priorM, priorU, 0.24, cluster = false, update = false)
@time astate, pM, pU, iter = penalized_likelihood_auction(pM, pU, compsum, priorM, priorU, 2.552, cluster = false, update = false)
pM
astate.nassigned

@time astate, pM, pU, iter = penalized_likelihood_auction(pM0, pU0, compsum, priorM, priorU, 0.0, cluster = true, update = false)
@time astate, pM, pU, iter = penalized_likelihood_auction(pM, pU, compsum, priorM, priorU, 0.24, cluster = true, update = false)
@time astate, pM, pU, iter = penalized_likelihood_auction(pM, pU, compsum, priorM, priorU, 2.552, cluster = true, update = false)
pM
astate.nassigned

@time astate, pM, pU, iter = penalized_likelihood_auction(pM0, pU0, compsum, priorM, priorU, 0.0, cluster = true, update = false)
@time astate, pM, pU, iter = penalized_likelihood_cluster_auction(pM0, pU0, compsum, priorM, priorU, 0.0)
@time astate, pM, pU, iter = penalized_likelihood_cluster_auction(pM, pU, compsum, priorM, priorU, 0.24)
@time astate, pM, pU, iter = penalized_likelihood_cluster_auction(pM, pU, compsum, priorM, priorU, 2.552)
pM
astate.nassigned

@time astate, pM, pU, iter = penalized_likelihood_auction(pM0, pU0, compsum, priorM, priorU, 0.0, cluster = false, update = true)
@time penalized_likelihood_auction_update(pM0, pU0, compsum, priorM, priorU, 0.0)
@time penalized_likelihood_auction_update(pM0, pU0, compsum, priorM, priorU, 0.24)
@time penalized_likelihood_auction_update(pM0, pU0, compsum, priorM, priorU, 2.552)
pM
astate.nassigned

@time astate, pM, pU, iter = penalized_likelihood_auction(pM0, pU0, compsum, priorM, priorU, 0.0, cluster = true, update = true)
@time penalized_likelihood_cluster_auction_update(pM0, pU0, compsum, priorM, priorU, 0.0)
@time penalized_likelihood_cluster_auction_update(pM0, pU0, compsum, priorM, priorU, 0.24)
@time penalized_likelihood_cluster_auction_update(pM0, pU0, compsum, priorM, priorU, 2.552)

@time pchain, penalties, iter = penalized_likelihood_search_hungarian(pM0, pU0, compsum, priorM, priorU, 0.0, 0.01, cluster = false)
pchain.nlinks
pchain.pM
@time pchain, penalties, iter = penalized_likelihood_search_hungarian(pM0, pU0, compsum, priorM, priorU, 0.0, 0.01, cluster = true)
pchain.nlinks
pchain.pM

@time pchain, penalties, iter = penalized_likelihood_search_auction(pM0, pU0, compsum, priorM, priorU, 0.0, 0.01, cluster = false, update = false)
pchain.nlinks
pchain.pM

@time pchain, penalties, iter = penalized_likelihood_search_auction(pM0, pU0, compsum, priorM, priorU, 0.0, 0.01, cluster = true,  update = false)
pchain.nlinks
pchain.pM

@time pchain, penalties, iter = penalized_likelihood_search_auction(pM0, pU0, compsum, priorM, priorU, 0.0, 0.01, cluster = false, update = true)
pchain.nlinks
pchain.pM

@time pchain, penalties, iter = penalized_likelihood_search_auction(pM0, pU0, compsum, priorM, priorU, 0.0, 0.01, cluster = true,  update = true)
pchain.nlinks
pchain.pM

########################################
#Test linkmatrix.jl
########################################

C1 = LinkMatrix(sparse([0,1,0,0]), sparse([2, 0, 0, 0]), 1, 4, 4)
C2 = LinkMatrix(sparse([0,1,0,0]), sparse([2, 0, 0, 0]))
C3 = LinkMatrix([0,1,0,0], [2, 0, 0, 0])
@test C1 == C2
@test C1 == C3
@test has_link(2,1,C1)
@test !has_link(1,2,C1)

@test add_link(3, 3, C1) != C1

row2col = sparse(Int8.([0, 1, 0, 0]))
col2row = sparse(Int8.([2, 0, 0, 0]))
