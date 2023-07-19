#include <nanobind/nanobind.h>

#include <pulp.h>

NB_MODULE(_pulp_ext_impl, m) {
    m.def("hello", []() { return "Hello world!"; });
}

