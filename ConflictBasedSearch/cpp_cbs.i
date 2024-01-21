%module cpp_cbs

%{
#include "cpp_cbs.h"
%}

%include "cpp_cbs.h"

// Expose the find_path function
extern std::vector<std::vector<Cell>> find_path();