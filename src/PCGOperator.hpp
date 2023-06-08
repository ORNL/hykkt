#pragma once

/**
 * @brief Interface for preconditioned operators, referred to as 'A' below
*/
class PCGOperator
{
public:
    virtual int get_operator_size() const = 0; //returns n dimension of nxn operator
    virtual void apply(double* v, double* out) = 0; //applies A to v st out = A v
    virtual void preconditioner_solve(double* b, double* out) = 0; //solves out = M^-1 b where M is the preconditioner for A
};