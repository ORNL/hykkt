using Test
using Random
using SparseArrays
using LinearAlgebra
using MatrixMarket


function writeVector(filename::String, vector)
    stream = open(filename, "w")
    write(stream, "%%MatrixMarket matrix\n")
    s = size(vector, 1)
    write(stream, "$(s) 1\n")
    for i in 1:s
        write(stream, "$(vector[i])\n")
    end
    close(stream)
end

function objective(Q, A, b, c, t, x)
    return t*(0.5*x'*Q*x + (c'*x)) - sum(log.(b-A*x))
end

function gradient(Q, A, b, c, t, x)
    return t*(Q*x + c) + A'*(1.0 ./ (b-A*x))
end

function hessian(A, b, x)
    return 1.0 ./ ((b-A*x).^2)
end


m = 10 # num constraints
n = 5 #num variables
rng = MersenneTwister(14324) # this is the seed
# These lines are from Boyd's example on how to generate an LP with a known solution 
# I'm using absolute values here to make sure all zeros is feasible (this is also realistic in bid optimization)
z = randn(rng,m)
zervec = zeros(m)
sst = max.(z, zervec)
y = sst-z
A = abs.(randn(rng,m,n))
foreach(normalize!,eachrow(A))
xst = abs.(randn(rng,n))
#xst2 = abs.(randn(rng,n))
b = A*xst+sst
#b2 = A*xst2+sst
c = -A'*y
# this generates the quadratic part of the problem
Q1 = sprand(rng, n, n, 1.0/n)
Q = Q1'*Q1 + I
A = sparse(A)
#vals1 = 1.0 ./ b .^ 2
#vals2 = 1.0 ./ b2 .^ 2

#H1 = Diagonal(vals1)
#H2 = Diagonal(vals2)
v_in = randn(rng,n)

t = 1.3

#S = Q + A' * H * A
#rows, cols, _ = findnz(Q)
#res = []
#for i in eachindex(rows)
#    push!(res, S[rows[i], cols[i]])
#end



#expected1 = (A' * (H1 * (A * v_in)))# + t * Q * v_in
#expected1 = sparse(expected1)
#expected2 = (A' * (H2 * (A * v_in)))# + t * Q * v_in
#expected2 = sparse(expected2)
#expected2 = sparse(res)
#expected3 = sparse(1.0 ./ diag(A' * H * A))
#H1 = sparse(vals1)
#H2 = sparse(vals2)
Q = sparse(Q)
#v_in = sparse(v_in)

#qp_objective_res = objective(Q, A, b, c, t, v_in)
#lp_objective_res = objective(0, A, b, c, t, v_in)
#qp_gradient_res = gradient(Q, A, b, c, t, v_in)
#lp_gradient_res = gradient(0, A, b, c, t, v_in)
#hessian_res = hessian(A, b, v_in)
#println(qp_objective_res)
#println(lp_objective_res)

curr = pwd() * "/src/mats/logbarrier"
q_filename = curr * "/q.mtx"
a_filename = curr * "/a.mtx"
#x_file_name = curr * "/x.mtx"
#qp_grad_file_name = curr * "/qp_gradient.mtx"
#lp_grad_file_name = curr * "/lp_gradient.mtx"
#inv_hess_file_name = curr * "/inv_hessian.mtx"
#vin_filename = curr * "/qp_vin_test.mtx"
#h1_filename = curr * "/lp_h1_test.mtx"
#h2_filename = curr * "/lp_h2_test.mtx"
b_file_name = curr * "/b.mtx"
c_file_name = curr * "/c.mtx"
#expected1_filename = curr * "/lp_res1_test.mtx"
#expected2_filename = curr * "/lp_res2_test.mtx"
#expected3_filename = curr * "/exp_diag3.mtx"

# Save the sparse matrix as an .mtx file
MatrixMarket.mmwrite(q_filename, Q);
MatrixMarket.mmwrite(a_filename, A);
writeVector(b_file_name, b)
writeVector(c_file_name, c)

#writeVector(x_file_name, v_in)
#writeVector(qp_grad_file_name, qp_gradient_res)
#writeVector(lp_grad_file_name, lp_gradient_res)
#writeVector(inv_hess_file_name, hessian_res)
#writeVector(h1_filename, H1)
#writeVector(h2_filename, H2)
#writeVector(vin_filename, v_in)
#writeVector(expected1_filename, expected1)
#writeVector(expected2_filename, expected2)
#writeVector(expected3_filename, expected3)