# DDCCMCKP
This is work related to paper, Chance-Constrained Multiple-Choice Knapsack Problem: Model, Algorithms, and Applications.

## Abstract
The multiple-choice knapsack problem (MCKP) is a classic NP-hard combinatorial optimization problem. 
Motivated by several significant real-world applications, this work investigates a novel variant of MCKP called chance-constrained multiple-choice knapsack problem (CCMCKP), where the item weights are random variables.
In particular, we focus on the practical scenario of CCMCKP, where the distribution of random weights is unknown but only sample data is available.
We first present the problem formulation of CCMCKP and then establish two benchmark sets.
The first set contains synthetic instances and the second set is devised to simulate a real-world application scenario of a certain telecommunication company.
To solve CCMCKP, we propose a data-driven adaptive local search (DDALS) algorithm.
The main merit of DDALS lies in its ability to evaluate solutions with chance constraints using data-driven methods, when only historical sample data is available and the underlying distributions are unknown.
Experimental results demonstrate the effectiveness of DDALS and show that it is superior to other baselines. 
Additionally, ablation studies confirm the necessity of each component in the algorithm.
Our proposed algorithm can serve as the baseline for future research, and the code and benchmark sets will be open-sourced to further promote research on this challenging problem.

##  supplementary materials.pdf
Due to space limitations, we have put the supporting materials and complete experimental results in the /supplementary.pdf for further understanding of our work. The supplementary materials (supplementary.pdf) contain complete experimental results, charts, and analysis involved in our research process. The purpose of these supporting materials is to provide readers with more detailed and comprehensive information for further understanding of our research work. In addition, these support materials can also help other researchers to replicate our experiments and verify our research results. We have uploaded this supplementary material to GitHub for free download and use by others.
