# Trotter-error-and-gate-complexity-of-the-SYK-and-sparse-SYK-models
Please find the source code here used for the numerical studies in the paper "Trotter error and gate complexity of the SYK and sparse SYK models" by Yiyuan Chen, Jonas Helsen and Maris Ozols (https://arxiv.org/abs/2502.18420). For questions regarding the code or the paper, please contact Yiyuan Chen at cyy020726@gmail.com. 

There are four simulation files (Simulation_Dense_n.py, Simulation_Dense_t.py, Simulation_Sparse_n.py, Simulation_Sparse_t.py). The 'Dense' files should be used to gather data for the dense SYK model, and the 'Sparse' files for the sparse SYK model. Use the '_n' files to compute the Trotter error at lower n's (n=6,...,18), and '_t' files to compute Trotter error from t=0 to t=1000. Please feel free to adjust these ranges. Our computational capability only allows us to compute up to n=18. Each simulation file utilizes a multiprocessing pipeline. You can adjust the number of cores used in the source code. Please be aware that the programs for n > 16 typically take a few hours to finish. The data is saved in the DATABASE folder and operates additively. 

For data analysis, you can find four analysis files (Analysis_Dense_n.py, Analysis_Dense_t.py, Analysis_Sparse_n.py, Analysis_Sparse_t.py). Each of them performs data analysis to the corresponding experiment. The '_n' analysis files compute the error ratio between the observed (normalized) Trotter error and our theoretical estimate (see paper). The '_t' analysis files plot the observed Trotter error and the theoretical bound with respect to time in two separate plots. The scaling exponent of the theoretical bounds is extracted by fitting a line ax+b through the log-log plot. 

Please make sure to have the following packages installed (via pip)
numpy
scipy
pandas
matplotlib
tqdm
