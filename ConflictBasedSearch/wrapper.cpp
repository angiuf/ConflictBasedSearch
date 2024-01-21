#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/chrono.h>
#include "HighLevelSolver.h"
#include "LowLevelSolver.h"
#include "util.h"
#include "cpp_cbs.h"

PYBIND11_MAKE_OPAQUE(std::vector<Cell>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<Cell>>);

namespace py = pybind11;

PYBIND11_MODULE(cpp_cbs, m) {
    py::class_<Cell>(m, "Cell")
    .def(py::init<>())
    .def(py::init<int, int>())
    .def_readwrite("isObstacle", &Cell::isObstacle)
    .def_readwrite("x", &Cell::x)
    .def_readwrite("y", &Cell::y)
    .def_readwrite("f", &Cell::f)
    .def_readwrite("g", &Cell::g)
    .def_readwrite("h", &Cell::h)
    .def("equal", &Cell::operator==)
    .def("assign", &Cell::operator=);


    py::bind_vector<std::vector<Cell>>(m, "VectorCell");
    py::bind_vector<std::vector<std::vector<Cell>>>(m, "VectorVectorCell");
    m.def("find_path", &find_path, "A function which finds a path");
}
