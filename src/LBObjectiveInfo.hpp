#pragma once
#include <cstddef>

struct LBObjectiveInfo
{
    double* out_buffer_d_ = nullptr;
    double out_h_ = 0.0; //change to a reference pointer and remove this pointer entirely
    void* storage_buffer_ = nullptr;
    size_t storage_bytes_ = 0;
    bool allocated_ = false;
};