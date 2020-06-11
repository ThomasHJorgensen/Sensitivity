This README file describes the Python code for replication of the Gourinchas and Parker (2002, ECMA) paper.
By Thomas H. Jørgensen, Contact: tjo@econ.ku.dk

Empirical files from the orignal paper is based on the online code for the paper by Andrews et al. (2017, QJE) and can be found here: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LLARSN

The data files are:
income.txt: 		age profile of average income (in 1000s of 1987 dollars)
sample_moments.txt: 	age profile of average consumption (in 1000s of 1987 dollars)
weight.txt: 		weight matrix used to estimate the structural parameters.

The replciation Python code is:
sens_GP2002.ipynb:  main code in Python Notebook. Run this with LOAD = FALSE in the top to re-estimate the parameters using the data described above. This script produces all figures and tables included in the paper.
solve_GP2002.py: this code solves the model using my Python implementation. This is called in the file above but can also be called by it self.

tools:
	ConsumptionSavingModel.py: 	basic consumption-class
	SimulatedMinimumDistance.py: 	SMD class used to estimate the model and calculate the sensitivity of the parameters in theta.
	plot.py: 			contains functions that plot the results and constructs tables.
	misc.py: 			contains Gauss-Hermite, linspace and other routines used to solve the model.
	linear_interp.py: 		linear interpolator constructed by Jeppe Druedahl (University of Copenhagen)