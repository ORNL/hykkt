using Random
using SparseArrays
using LinearAlgebra
using MatrixMarket
using CUDA
using CUDA.CUSPARSE
using Test
export CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseVector

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

m = 25000
n = 10000
rng = MersenneTwister(14324) # this is the seed

for i in 1:10
    A = sprand(rng, Float64,m,n,0.01)
    x = rand(rng, Float64,n)
    y = rand(Float64,m)
    alpha = rand(Float64)
    beta = rand(Float64)
    d_A = CuSparseMatrixCSR(A)
    d_x = CuArray(x)
    d_y = CuArray(y)
    @time CUSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
end
exit()


#A = sprand(rng, m, n, 0.01)
#x = randn(rng, n)

#lst = []
#for i in 1:10
#    push!(lst, (CuSparseMatrixCSR{Float64}(sprand(rng, m, n, 0.01)), CuVector{Float64}(randn(rng,n))))
#end

y = A * x
#for pair in lst
#    y_c = @time pair[1] * pair[2]
#end
A = sparse(A)
x = sparse(x)
y = sparse(y)

curr = pwd() * "/src/mats/sandbox"
a_filename = curr * "/a.mtx"
x_filename = curr * "/x.mtx"
y_filename = curr * "/y.mtx"

MatrixMarket.mmwrite(a_filename, A)
writeVector(x_filename, x)
writeVector(y_filename, y)