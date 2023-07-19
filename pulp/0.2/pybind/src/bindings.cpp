#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <pulp.h>

#include <cstdio>

namespace nb = nanobind;

NB_MODULE(_pulp_ext_impl, m) {
  m.def("hello", []() { return "Hello world!"; });

  m.def(
      "partition_graph",
      [](nb::ndarray<> a, int num_parts) {
        printf("Array data pointer : %p\n", a.data());
        printf("Array dimension : %zu\n", a.ndim());
        for (size_t i = 0; i < a.ndim(); ++i) {
          printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
          printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
        }
        printf("Device ID = %u (cpu=%i, cuda=%i)\n", a.device_id(),
               int(a.device_type() == nb::device::cpu::value),
               int(a.device_type() == nb::device::cuda::value));
        printf("Array dtype: int16=%i, uint32=%i, float32=%i\n",
               a.dtype() == nb::dtype<int16_t>(),
               a.dtype() == nb::dtype<uint32_t>(),
               a.dtype() == nb::dtype<float>());

        // return pulp_run(g, ppc, parts, num_parts);
      },
      nb::arg("vtx_indptr"), nb::arg("num_parts"), "Partition a graph.");
}
