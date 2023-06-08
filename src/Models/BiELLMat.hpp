#pragma once

template<typename T>
class BiELLMat 
{
public:
    BiELLMat(int* csr_i, int* csr_j, T* csr_v)
    {

    }
    ~BiELLMat()
    {

    }
private:
    T* a_;
    T* ja_;
    int* ia_;
    int* perm_;
};