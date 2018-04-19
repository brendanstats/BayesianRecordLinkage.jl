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
        end
    end

end

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
