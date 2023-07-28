
#include <memory>
#include <optional>
#include <stdexcept>
#include <cstdio>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <pulp.h>
namespace nb = nanobind;

std::shared_ptr<pulp_graph_t> graph_py_to_cpp(
    nb::ndarray<> out_array,
    nb::ndarray<> out_degree_list,
    std::optional<nb::ndarray<>> vertex_weights,
    std::optional<nb::ndarray<>> edge_weights,
)
{
  auto graph = std::make_shared<pulp_graph_t>();

  if(out_array.dtype() != nb.dtype<int>()) {
    throw std::runtime_error("out_array must be int dtype");
  }

  if(out_degree_list.dtype() != nb.dtype<long>()) {
    throw std::runtime_error("out_degree_list must be long dtype");
  }


  return graph;
}

NB_MODULE(_pulp_ext_impl, m) {
  m.def("hello", []() { return "Hello world!"; });
  m.def(
      "partition_graph",
      [](nb::ndarray<> a) {

  m.def(
      "inspect",
      [](nb::ndarray<> a) {
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

      },
      nb::arg("array"),  "Inspect an array.");
}
