#compiler flags
NVCC = nvcc -ccbin=gcc -Xcompiler -g -G -gencode arch=compute_80,code=sm_80 -DCUDA11
NVCC_FLAGS = -lcusparse -lcusolver -lcublas -lrt -lgfortran

#dependencies for the hybrid solver, permutation alone, and Ruiz scaling alone
OBJS = cuSolver_driver_hybrid.o adapted_Ruiz_scaling.o Ruiz_scaling.o input_functions.o permcheck.o schur_complement_cg.o  
P_OBJS = perm_driver.o permcheck.o 
R_OBJS = Ruiz_driver.o adapted_Ruiz_scaling.o Ruiz_scaling.o 

RM = rm
.PHONY: clean

hybrid_solver: $(OBJS) $(NVCC) -o hybrid_solver $(NVCC_FLAGS) $(OBJS)
run_perm: $(OBJS) $(NVCC) -o run_perm $(NVCC_FLAGS) $(OBJS)
run_Ruiz: $(OBJS) $(NVCC) -o run_Ruiz $(NVCC_FLAGS) $(OBJS)

%.o : %.cu
		$(NVCC) $(NVCC_FLAGS) -c $< 

# Specify dependencies
cuSolver_driver_hybrid.o : cuSolver_driver_hybrid.cu adapted_Ruiz_scaling.cu Ruiz_scaling.cu input_functions.cu permcheck.cu schur_complement_cg.cu 
perm_driver.o : perm_driver.cu permcheck.cu
Ruiz_driver.o : Ruiz_driver.cu adapted_Ruiz_scaling.cu Ruiz_scaling.cu
input_functions.o : input_functions.cu  
permcheck.o : permcheck.cu
Ruiz_scaling.o : Ruiz_scaling.cu
schur_complement.o : schur_complement.cu 
adapted_Ruiz_scaling.o : adapted_Ruiz_scaling.cu

clean: -$(RM) *.o hybrid_solver run_perm run_Ruiz
