
#include "quantum.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <tuple>

namespace py = pybind11;
using namespace block2;

PYBIND11_MAKE_OPAQUE(vector<int>);
PYBIND11_MAKE_OPAQUE(vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(vector<uint16_t>);
PYBIND11_MAKE_OPAQUE(vector<double>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpExpr>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpString>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement>>);
PYBIND11_MAKE_OPAQUE(vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrixInfo>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Symbolic>>);
PYBIND11_MAKE_OPAQUE(map<OpNames, shared_ptr<SparseMatrix>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition>>);
PYBIND11_MAKE_OPAQUE(
    map<shared_ptr<OpExpr>, shared_ptr<SparseMatrix>, op_expr_less>);

template <typename T> struct Array {
    T *data;
    size_t n;
    Array(T *data, size_t n) : data(data), n(n) {}
    T &operator[](size_t idx) { return data[idx]; }
};

template <typename T, typename... Args>
size_t printable(const T &p, const Args &...) {
    return (size_t)&p;
}

template <typename T>
auto printable(const T &p)
    -> decltype(declval<ostream &>() << declval<T>(), T()) {
    return p;
}

template <typename T>
py::class_<Array<T>, shared_ptr<Array<T>>> bind_array(py::module &m,
                                                      const char *name) {
    return py::class_<Array<T>, shared_ptr<Array<T>>>(m, name)
        .def("__setitem__", [](Array<T> *self, size_t i,
                               const T &t) { self->operator[](i) = t; })
        .def("__getitem__",
             [](Array<T> *self, size_t i) { return self->operator[](i); })
        .def("__len__", [](Array<T> *self) { return self->n; })
        .def("__repr__",
             [name](Array<T> *self) {
                 stringstream ss;
                 ss << name << "(LEN=" << self->n << ")[";
                 for (size_t i = 0; i < self->n; i++)
                     ss << printable<T>(self->data[i]) << ",";
                 ss << "]";
                 return ss.str();
             })
        .def(
            "__iter__",
            [](Array<T> *self) {
                return py::make_iterator<
                    py::return_value_policy::reference_internal, T *, T *, T &>(
                    self->data, self->data + self->n);
            },
            py::keep_alive<0, 1>());
}

PYBIND11_MODULE(block2, m) {

    m.doc() = "python interface for block 2.0.";

    py::bind_vector<vector<int>>(m, "VectorInt");
    py::bind_vector<vector<uint16_t>>(m, "VectorUInt16");
    py::bind_vector<vector<double>>(m, "VectorDouble");
    py::bind_vector<vector<uint8_t>>(m, "VectorUInt8")
        .def_property_readonly(
            "ptr",
            [](vector<uint8_t> *self) -> uint8_t * { return &(*self)[0]; })
        .def("__str__", [](vector<uint8_t> *self) {
            stringstream ss;
            ss << "VectorUInt8[ ";
            for (auto p : *self)
                ss << (int)p << " ";
            ss << "]";
            return ss.str();
        });

    m.def(
        "init_memory",
        [](size_t isize, size_t dsize, const string &save_dir) {
            frame = new DataFrame(isize, dsize, save_dir);
        },
        py::arg("isize") = size_t(1L << 28),
        py::arg("dsize") = size_t(1L << 30), py::arg("save_dir") = "nodex");

    m.def("release_memory", []() {
        frame->activate(0);
        assert(ialloc->used == 0 && dalloc->used == 0);
        delete frame;
    });

    py::class_<StackAllocator<uint32_t>>(m, "IntAllocator")
        .def(py::init<>())
        .def_readwrite("size", &StackAllocator<uint32_t>::size)
        .def_readwrite("used", &StackAllocator<uint32_t>::used)
        .def_readwrite("shift", &StackAllocator<uint32_t>::shift);

    py::class_<StackAllocator<double>>(m, "DoubleAllocator")
        .def(py::init<>())
        .def_readwrite("size", &StackAllocator<double>::size)
        .def_readwrite("used", &StackAllocator<double>::used)
        .def_readwrite("shift", &StackAllocator<double>::shift);

    struct Global {};

    py::class_<DataFrame, shared_ptr<DataFrame>>(m, "DataFrame")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, const string &>())
        .def_readwrite("save_dir", &DataFrame::save_dir)
        .def_readwrite("prefix", &DataFrame::prefix)
        .def_readwrite("isize", &DataFrame::isize)
        .def_readwrite("dsize", &DataFrame::dsize)
        .def_readwrite("n_frames", &DataFrame::n_frames)
        .def_readwrite("i_frame", &DataFrame::i_frame)
        .def_readwrite("iallocs", &DataFrame::iallocs)
        .def_readwrite("dallocs", &DataFrame::dallocs)
        .def("activate", &DataFrame::activate)
        .def("load_data", &DataFrame::load_data)
        .def("save_data", &DataFrame::save_data)
        .def("reset", &DataFrame::reset);

    py::class_<Global>(m, "Global")
        .def_property_static(
            "ialloc", [](py::object) { return ialloc; },
            [](py::object, StackAllocator<uint32_t> *ia) { ialloc = ia; })
        .def_property_static(
            "dalloc", [](py::object) { return dalloc; },
            [](py::object, StackAllocator<double> *da) { dalloc = da; })
        .def_property_static(
            "frame", [](py::object) { return frame; },
            [](py::object, DataFrame *fr) { frame = fr; });

    bind_array<uint8_t>(m, "ArrayUInt8")
        .def("__str__", [](Array<uint8_t> *self) {
            stringstream ss;
            ss << "ArrayUInt8(LEN=" << self->n << ")[ ";
            for (size_t i = 0; i < self->n; i++)
                ss << (int)self->data[i] << " ";
            ss << "]";
            return ss.str();
        });
    bind_array<uint16_t>(m, "ArrayUInt16");
    bind_array<StateInfo>(m, "ArrayStateInfo");
    bind_array<SpinLabel>(m, "ArraySpinLabel");
    bind_array<vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>>>(
        m, "ArrayVectorPLMatInfo");

    py::class_<Random, shared_ptr<Random>>(m, "Random")
        .def_static("rand_seed", &Random::rand_seed, py::arg("i") = 0U)
        .def_static("rand_int", &Random::rand_int, py::arg("a"), py::arg("b"))
        .def_static("rand_double", &Random::rand_double, py::arg("a") = 0,
                    py::arg("b") = 1)
        .def_static("fill_rand_double",
                    [](py::object, py::array_t<double> &data, double a = 0,
                       double b = 1) {
                        return Random::fill_rand_double(data.mutable_data(),
                                                        data.size(), a, b);
                    });

    py::class_<MatrixRef, shared_ptr<MatrixRef>>(m, "Matrix",
                                                 py::buffer_protocol())
        .def_buffer([](MatrixRef *self) -> py::buffer_info {
            return py::buffer_info(self->data, sizeof(double),
                                   py::format_descriptor<double>::format(), 2,
                                   {self->m, self->n},
                                   {sizeof(double) * self->n, sizeof(double)});
        });

    py::class_<MatrixFunctions>(m, "MatrixFunctions")
        .def_static(
            "eigs", [](int n, py::array_t<double> &a, py::array_t<double> &w) {
                MatrixFunctions::eigs(MatrixRef(a.mutable_data(), n, n),
                                      DiagonalMatrix(w.mutable_data(), n));
            });

    py::class_<DiagonalMatrix, shared_ptr<DiagonalMatrix>>(
        m, "DiagonalMatrix", py::buffer_protocol())
        .def_buffer([](DiagonalMatrix *self) -> py::buffer_info {
            return py::buffer_info(self->data, sizeof(double),
                                   py::format_descriptor<double>::format(), 1,
                                   {self->n}, {sizeof(double)});
        });

    py::class_<FCIDUMP, shared_ptr<FCIDUMP>>(m, "FCIDUMP")
        .def(py::init<>())
        .def("read", &FCIDUMP::read)
        .def("deallocate", &FCIDUMP::deallocate)
        .def_property_readonly("orb_sym", &FCIDUMP::orb_sym)
        .def_property_readonly("n_elec", &FCIDUMP::n_elec)
        .def_property_readonly("twos", &FCIDUMP::twos)
        .def_property_readonly("isym", &FCIDUMP::isym)
        .def_property_readonly("n_sites", &FCIDUMP::n_sites)
        .def_readwrite("params", &FCIDUMP::params)
        .def_readwrite("ts", &FCIDUMP::ts)
        .def_readwrite("vs", &FCIDUMP::vs)
        .def_readwrite("vabs", &FCIDUMP::vabs)
        .def_readwrite("e", &FCIDUMP::e)
        .def_readwrite("total_memory", &FCIDUMP::total_memory)
        .def_readwrite("uhf", &FCIDUMP::uhf);

    py::class_<SpinLabel>(m, "SpinLabel")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, int>())
        .def_readwrite("data", &SpinLabel::data)
        .def_property("n", &SpinLabel::n, &SpinLabel::set_n)
        .def_property("twos", &SpinLabel::twos, &SpinLabel::set_twos)
        .def_property("twos_low", &SpinLabel::twos_low,
                      &SpinLabel::set_twos_low)
        .def_property("pg", &SpinLabel::pg, &SpinLabel::set_pg)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SpinLabel::get_ket)
        .def("get_bra", &SpinLabel::get_bra, py::arg("dq"))
        .def("__repr__", &SpinLabel::to_str);

    py::enum_<OpNames>(m, "OpNames", py::arithmetic())
        .value("H", OpNames::H)
        .value("I", OpNames::I)
        .value("N", OpNames::N)
        .value("NN", OpNames::NN)
        .value("NUD", OpNames::NUD)
        .value("C", OpNames::C)
        .value("D", OpNames::D)
        .value("R", OpNames::R)
        .value("RD", OpNames::RD)
        .value("A", OpNames::A)
        .value("AD", OpNames::AD)
        .value("P", OpNames::P)
        .value("PD", OpNames::PD)
        .value("B", OpNames::B)
        .value("Q", OpNames::Q)
        .value("Zero", OpNames::Zero)
        .value("PDM1", OpNames::PDM1);

    py::enum_<OpTypes>(m, "OpTypes", py::arithmetic())
        .value("Zero", OpTypes::Zero)
        .value("Elem", OpTypes::Elem)
        .value("Prod", OpTypes::Prod)
        .value("Sum", OpTypes::Sum)
        .value("ElemRef", OpTypes::ElemRef)
        .value("SumProd", OpTypes::SumProd);

    py::class_<OpExpr, shared_ptr<OpExpr>>(m, "OpExpr")
        .def(py::init<>())
        .def("get_type", &OpExpr::get_type)
        .def(py::self == py::self)
        .def("__repr__", &to_str);

    py::class_<SiteIndex>(m, "SiteIndex")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<uint8_t>())
        .def(py::init<uint8_t, uint8_t, uint8_t>())
        .def("size", &SiteIndex::size)
        .def("spin_size", &SiteIndex::spin_size)
        .def("s", &SiteIndex::s, py::arg("i") = 0)
        .def("__getitem__",
             [](SiteIndex *self, uint8_t i) { return self->operator[](i); })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def("__hash__", &SiteIndex::hash)
        .def("__repr__", &SiteIndex::to_str);

    py::class_<OpElement, shared_ptr<OpElement>, OpExpr>(m, "OpElement")
        .def(py::init<OpNames, SiteIndex, SpinLabel>())
        .def(py::init<OpNames, SiteIndex, SpinLabel, double>())
        .def_readwrite("name", &OpElement::name)
        .def_readwrite("site_index", &OpElement::site_index)
        .def_readwrite("factor", &OpElement::factor)
        .def_readwrite("q_label", &OpElement::q_label)
        .def("abs", &OpElement::abs)
        .def("__mul__", &OpElement::operator*)
        .def(py::self == py::self)
        .def(py::self < py::self)
        .def("__hash__", &OpElement::hash);

    py::class_<OpElementRef, shared_ptr<OpElementRef>, OpExpr>(m,
                                                               "OpElementRef")
        .def(py::init<const shared_ptr<OpElement> &, int8_t, int8_t>())
        .def_readwrite("op", &OpElementRef::op)
        .def_readwrite("factor", &OpElementRef::factor)
        .def_readwrite("trans", &OpElementRef::trans);

    py::class_<OpString, shared_ptr<OpString>, OpExpr>(m, "OpString")
        .def(py::init<const shared_ptr<OpElement> &, double>())
        .def(py::init<const shared_ptr<OpElement> &, double, uint8_t>())
        .def(py::init<const shared_ptr<OpElement> &,
                      const shared_ptr<OpElement> &, double>())
        .def(py::init<const shared_ptr<OpElement> &,
                      const shared_ptr<OpElement> &, double, uint8_t>())
        .def_readwrite("factor", &OpString::factor)
        .def_readwrite("conj", &OpString::conj)
        .def_readwrite("a", &OpString::a)
        .def_readwrite("b", &OpString::b);

    py::class_<OpSumProd, shared_ptr<OpSumProd>, OpString>(m, "OpSumProd")
        .def(py::init<const shared_ptr<OpElement> &,
                      const vector<shared_ptr<OpElement>> &,
                      const vector<bool> &, double, uint8_t>())
        .def(py::init<const shared_ptr<OpElement> &,
                      const vector<shared_ptr<OpElement>> &,
                      const vector<bool> &, double>())
        .def(py::init<const vector<shared_ptr<OpElement>> &,
                      const shared_ptr<OpElement> &, const vector<bool> &,
                      double, uint8_t>())
        .def(py::init<const vector<shared_ptr<OpElement>> &,
                      const shared_ptr<OpElement> &, const vector<bool> &,
                      double>())
        .def_readwrite("ops", &OpSumProd::ops)
        .def_readwrite("conjs", &OpSumProd::conjs);

    py::class_<OpSum, shared_ptr<OpSum>, OpExpr>(m, "OpSum")
        .def(py::init<const vector<shared_ptr<OpString>> &>())
        .def_readwrite("strings", &OpSum::strings);

    py::bind_vector<vector<shared_ptr<OpExpr>>>(m, "VectorOpExpr");
    py::bind_vector<vector<shared_ptr<OpElement>>>(m, "VectorOpElement");
    py::bind_vector<vector<shared_ptr<OpString>>>(m, "VectorOpString");

    py::enum_<SymTypes>(m, "SymTypes", py::arithmetic())
        .value("RVec", SymTypes::RVec)
        .value("CVec", SymTypes::CVec)
        .value("Mat", SymTypes::Mat);

    py::class_<Symbolic, shared_ptr<Symbolic>>(m, "Symbolic")
        .def_readwrite("m", &Symbolic::m)
        .def_readwrite("n", &Symbolic::n)
        .def_readwrite("data", &Symbolic::data)
        .def("get_type", [](Symbolic *self) { return self->get_type(); })
        .def("__matmul__",
             [](const shared_ptr<Symbolic> &self,
                const shared_ptr<Symbolic> &other) { return self * other; });

    py::class_<StateInfo, shared_ptr<StateInfo>>(m, "StateInfo")
        .def_readwrite("n", &StateInfo::n)
        .def_readwrite("n_states_total", &StateInfo::n_states_total)
        .def_property_readonly("quanta",
                               [](StateInfo *self) {
                                   return Array<SpinLabel>(self->quanta,
                                                           self->n);
                               })
        .def_property_readonly("n_states",
                               [](StateInfo *self) {
                                   return Array<uint16_t>(self->n_states,
                                                          self->n);
                               })
        .def("deallocate", &StateInfo::deallocate)
        .def("__repr__", [](StateInfo *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<SparseMatrixInfo::ConnectionInfo,
               shared_ptr<SparseMatrixInfo::ConnectionInfo>>(m,
                                                             "ConnectionInfo")
        .def(py::init<>())
        .def_property_readonly(
            "n",
            [](SparseMatrixInfo::ConnectionInfo *self) { return self->n[4]; })
        .def_readwrite("nc", &SparseMatrixInfo::ConnectionInfo::nc)
        .def("deallocate", &SparseMatrixInfo::ConnectionInfo::deallocate)
        .def("__repr__", [](SparseMatrixInfo::ConnectionInfo *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<SparseMatrixInfo, shared_ptr<SparseMatrixInfo>>(
        m, "SparseMatrixInfo")
        .def(py::init<>())
        .def_readwrite("delta_quantum", &SparseMatrixInfo::delta_quantum)
        .def_readwrite("is_fermion", &SparseMatrixInfo::is_fermion)
        .def_readwrite("is_wavefunction", &SparseMatrixInfo::is_wavefunction)
        .def_readwrite("n", &SparseMatrixInfo::n)
        .def_readwrite("cinfo", &SparseMatrixInfo::cinfo)
        .def_property_readonly("quanta",
                               [](SparseMatrixInfo *self) {
                                   return Array<SpinLabel>(self->quanta,
                                                           self->n);
                               })
        .def_property_readonly("n_states_total",
                               [](SparseMatrixInfo *self) {
                                   return py::array_t<uint32_t>(
                                       self->n, self->n_states_total);
                               })
        .def_property_readonly("n_states_bra",
                               [](SparseMatrixInfo *self) {
                                   return py::array_t<uint16_t>(
                                       self->n, self->n_states_bra);
                               })
        .def_property_readonly("n_states_ket",
                               [](SparseMatrixInfo *self) {
                                   return py::array_t<uint16_t>(
                                       self->n, self->n_states_ket);
                               })
        .def("initialize", &SparseMatrixInfo::initialize, py::arg("bra"),
             py::arg("ket"), py::arg("dq"), py::arg("is_fermion"),
             py::arg("wfn") = false)
        .def("find_state", &SparseMatrixInfo::find_state, py::arg("q"),
             py::arg("start") = 0)
        .def_property_readonly("total_memory",
                               &SparseMatrixInfo::get_total_memory)
        .def("allocate", &SparseMatrixInfo::allocate, py::arg("length"),
             py::arg("ptr") = nullptr)
        .def("deallocate", &SparseMatrixInfo::deallocate)
        .def("reallocate", &SparseMatrixInfo::reallocate, py::arg("length"))
        .def("__repr__", [](SparseMatrixInfo *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<SparseMatrix, shared_ptr<SparseMatrix>>(m, "SparseMatrix")
        .def(py::init<>())
        .def_readwrite("info", &SparseMatrix::info)
        .def_readwrite("factor", &SparseMatrix::factor)
        .def_readwrite("total_memory", &SparseMatrix::total_memory)
        .def_property_readonly("data",
                               [](SparseMatrix *self) {
                                   return py::array_t<double>(
                                       self->total_memory, self->data);
                               })
        .def("clear", &SparseMatrix::clear)
        .def("load_data", &SparseMatrix::load_data, py::arg("filename"),
             py::arg("load_info") = false)
        .def("save_data", &SparseMatrix::save_data, py::arg("filename"),
             py::arg("save_info") = false)
        .def("copy_data_from", &SparseMatrix::copy_data_from)
        .def("allocate",
             [](SparseMatrix *self, const shared_ptr<SparseMatrixInfo> &info) {
                 self->allocate(info);
             })
        .def("deallocate", &SparseMatrix::deallocate)
        .def("reallocate", &SparseMatrix::reallocate, py::arg("length"))
        .def("trace", &SparseMatrix::trace)
        .def("norm", &SparseMatrix::norm)
        .def("__getitem__",
             [](SparseMatrix *self, int idx) { return (*self)[idx]; })
        .def("left_canonicalize", &SparseMatrix::left_canonicalize,
             py::arg("rmat"))
        .def("right_canonicalize", &SparseMatrix::right_canonicalize,
             py::arg("lmat"))
        .def("left_multiply", &SparseMatrix::left_multiply, py::arg("lmat"),
             py::arg("l"), py::arg("m"), py::arg("r"), py::arg("lm"),
             py::arg("lm_cinfo"))
        .def("right_multiply", &SparseMatrix::right_multiply, py::arg("rmat"),
             py::arg("l"), py::arg("m"), py::arg("r"), py::arg("mr"),
             py::arg("mr_cinfo"))
        .def("randomize", &SparseMatrix::randomize, py::arg("a") = 0.0,
             py::arg("b") = 1.0)
        .def("contract", &SparseMatrix::contract)
        .def("swap_to_fused_left", &SparseMatrix::swap_to_fused_left)
        .def("swap_to_fused_right", &SparseMatrix::swap_to_fused_right)
        .def("__repr__", [](SparseMatrix *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>>>(
        m, "VectorPLMatInfo");
    py::bind_vector<vector<shared_ptr<SparseMatrixInfo>>>(m, "VectorSpMatInfo");
    py::bind_vector<vector<shared_ptr<SparseMatrix>>>(m, "VectorSpMat");
    py::bind_map<map<OpNames, shared_ptr<SparseMatrix>>>(m, "MapOpNamesSpMat");
    py::bind_map<
        map<shared_ptr<OpExpr>, shared_ptr<SparseMatrix>, op_expr_less>>(
        m, "MapOpExprSpMat");

    py::enum_<AncillaTypes>(m, "AncillaTypes", py::arithmetic())
        .value("Nothing", AncillaTypes::None)
        .value("Ancilla", AncillaTypes::Ancilla);

    py::class_<MPSInfo, shared_ptr<MPSInfo>>(m, "MPSInfo")
        .def_readwrite("n_sites", &MPSInfo::n_sites)
        .def_readwrite("vaccum", &MPSInfo::vaccum)
        .def_readwrite("target", &MPSInfo::target)
        .def_readwrite("orbsym", &MPSInfo::orbsym)
        .def_readwrite("n_syms", &MPSInfo::n_syms)
        .def_readwrite("bond_dim", &MPSInfo::bond_dim)
        .def_readwrite("tag", &MPSInfo::tag)
        .def(py::init([](int n_sites, SpinLabel vaccum, SpinLabel target,
                         Array<StateInfo> &basis, const vector<uint8_t> &orbsym,
                         uint8_t n_syms) {
            return make_shared<MPSInfo>(n_sites, vaccum, target, basis.data,
                                        orbsym, n_syms);
        }))
        .def_property_readonly("basis",
                               [](MPSInfo *self) {
                                   return Array<StateInfo>(self->basis,
                                                           self->n_syms);
                               })
        .def_property_readonly("left_dims_fci",
                               [](MPSInfo *self) {
                                   return Array<StateInfo>(self->left_dims_fci,
                                                           self->n_sites + 1);
                               })
        .def_property_readonly("right_dims_fci",
                               [](MPSInfo *self) {
                                   return Array<StateInfo>(self->right_dims_fci,
                                                           self->n_sites + 1);
                               })
        .def_property_readonly("left_dims",
                               [](MPSInfo *self) {
                                   return Array<StateInfo>(self->left_dims,
                                                           self->n_sites + 1);
                               })
        .def_property_readonly("right_dims",
                               [](MPSInfo *self) {
                                   return Array<StateInfo>(self->right_dims,
                                                           self->n_sites + 1);
                               })
        .def("get_ancilla_type", &MPSInfo::get_ancilla_type)
        .def("set_bond_dimension_using_occ",
             &MPSInfo::set_bond_dimension_using_occ, py::arg("m"),
             py::arg("occ"), py::arg("bias") = 1.0)
        .def("set_bond_dimension", &MPSInfo::set_bond_dimension)
        .def("get_filename", &MPSInfo::get_filename)
        .def("save_mutable", &MPSInfo::save_mutable)
        .def("deallocate_mutable", &MPSInfo::deallocate_mutable)
        .def("load_mutable", &MPSInfo::load_mutable)
        .def("save_left_dims", &MPSInfo::save_left_dims)
        .def("save_right_dims", &MPSInfo::save_right_dims)
        .def("load_left_dims", &MPSInfo::load_left_dims)
        .def("load_right_dims", &MPSInfo::load_right_dims)
        .def("deallocate", &MPSInfo::deallocate);

    py::class_<AncillaMPSInfo, shared_ptr<AncillaMPSInfo>, MPSInfo>(
        m, "AncillaMPSInfo")
        .def_readwrite("n_physical_sites", &AncillaMPSInfo::n_physical_sites)
        .def(py::init([](int n_sites, SpinLabel vaccum, SpinLabel target,
                         Array<StateInfo> &basis, const vector<uint8_t> &orbsym,
                         uint8_t n_syms) {
            return make_shared<AncillaMPSInfo>(n_sites, vaccum, target,
                                               basis.data, orbsym, n_syms);
        }))
        .def_static("trans_orbsym", &AncillaMPSInfo::trans_orbsym)
        .def("set_thermal_limit", &AncillaMPSInfo::set_thermal_limit);

    py::class_<MPS, shared_ptr<MPS>>(m, "MPS")
        .def(py::init<int, int, int>())
        .def_readwrite("n_sites", &MPS::n_sites)
        .def_readwrite("center", &MPS::center)
        .def_readwrite("dot", &MPS::dot)
        .def_readwrite("info", &MPS::info)
        .def_readwrite("tensors", &MPS::tensors)
        .def_readwrite("canonical_form", &MPS::canonical_form)
        .def("initialize", &MPS::initialize)
        .def("fill_thermal_limit", &MPS::fill_thermal_limit)
        .def("canonicalize", &MPS::canonicalize)
        .def("random_canonicalize", &MPS::random_canonicalize)
        .def("get_filename", &MPS::get_filename)
        .def("save_mutable", &MPS::save_mutable)
        .def("save_tensor", &MPS::save_tensor)
        .def("load_tensor", &MPS::load_tensor)
        .def("unload_tensor", &MPS::unload_tensor)
        .def("deallocate", &MPS::deallocate);

    py::enum_<SeqTypes>(m, "SeqTypes", py::arithmetic())
        .value("Nothing", SeqTypes::None)
        .value("Simple", SeqTypes::Simple)
        .value("Auto", SeqTypes::Auto);

    py::class_<BatchGEMMSeq, shared_ptr<BatchGEMMSeq>>(m, "BatchGEMMSeq")
        .def_readwrite("batch", &BatchGEMMSeq::batch)
        .def_readwrite("post_batch", &BatchGEMMSeq::post_batch)
        .def_readwrite("refs", &BatchGEMMSeq::refs)
        .def_readwrite("cumulative_nflop", &BatchGEMMSeq::cumulative_nflop)
        .def_readwrite("mode", &BatchGEMMSeq::mode)
        .def(py::init<>())
        .def(py::init<size_t>())
        .def(py::init<size_t, SeqTypes>())
        .def("iadd", &BatchGEMMSeq::iadd, py::arg("a"), py::arg("b"),
             py::arg("scale") = 1.0, py::arg("cfactor") = 1.0,
             py::arg("conj") = false)
        .def("rotate", &BatchGEMMSeq::rotate, py::arg("a"), py::arg("c"),
             py::arg("bra"), py::arg("conj_bra"), py::arg("ket"),
             py::arg("conj_ket"), py::arg("scale"))
        .def("tensor_product_diagonal", &BatchGEMMSeq::tensor_product_diagonal,
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("scale"))
        .def("tensor_product", &BatchGEMMSeq::tensor_product, py::arg("a"),
             py::arg("conja"), py::arg("b"), py::arg("conjb"), py::arg("c"),
             py::arg("scale"), py::arg("stride"))
        .def("divide_batch", &BatchGEMMSeq::divide_batch)
        .def("check", &BatchGEMMSeq::check)
        .def("prepare", &BatchGEMMSeq::prepare)
        .def("allocate", &BatchGEMMSeq::allocate)
        .def("deallocate", &BatchGEMMSeq::deallocate)
        .def("simple_perform", &BatchGEMMSeq::simple_perform)
        .def("auto_perform", &BatchGEMMSeq::auto_perform)
        .def("perform", &BatchGEMMSeq::perform)
        .def("clear", &BatchGEMMSeq::clear)
        .def("__call__", &BatchGEMMSeq::operator())
        .def("__repr__", [](BatchGEMMSeq *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<CG, shared_ptr<CG>>(m, "CG")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def("initialize", &CG::initialize)
        .def("deallocate", &CG::deallocate)
        .def_static("triangle", &CG::triangle, py::arg("tja"), py::arg("tjb"),
                    py::arg("tjc"))
        .def("sqrt_delta", &CG::sqrt_delta, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"))
        .def("cg", &CG::cg, py::arg("tja"), py::arg("tjb"), py::arg("tjc"),
             py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_3j", &CG::wigner_3j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_6j", &CG::wigner_6j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"))
        .def("wigner_9j", &CG::wigner_9j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"),
             py::arg("tjg"), py::arg("tjh"), py::arg("tji"))
        .def("racah", &CG::racah, py::arg("ta"), py::arg("tb"), py::arg("tc"),
             py::arg("td"), py::arg("te"), py::arg("tf"))
        .def("transpose_cg", &CG::transpose_cg, py::arg("td"), py::arg("tl"),
             py::arg("tr"));

    py::class_<OperatorFunctions, shared_ptr<OperatorFunctions>>(
        m, "OperatorFunctions")
        .def_readwrite("cg", &OperatorFunctions::cg)
        .def_readwrite("seq", &OperatorFunctions::seq)
        .def(py::init<const shared_ptr<CG> &>())
        .def("iadd", &OperatorFunctions::iadd, py::arg("a"), py::arg("b"),
             py::arg("scale") = 1.0, py::arg("conj") = false)
        .def("tensor_rotate", &OperatorFunctions::tensor_rotate, py::arg("a"),
             py::arg("c"), py::arg("rot_bra"), py::arg("rot_ket"),
             py::arg("trans"), py::arg("scale") = 1.0)
        .def("tensor_product_diagonal",
             &OperatorFunctions::tensor_product_diagonal, py::arg("conj"),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("opdq"),
             py::arg("scale") = 1.0)
        .def("tensor_product_multiply",
             &OperatorFunctions::tensor_product_multiply, py::arg("conj"),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("v"),
             py::arg("opdq"), py::arg("scale") = 1.0)
        .def("tensor_product", &OperatorFunctions::tensor_product,
             py::arg("conj"), py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("scale") = 1.0)
        .def("product", &OperatorFunctions::product, py::arg("a"), py::arg("b"),
             py::arg("c"), py::arg("scale") = 1.0)
        .def_static("trans_product", &OperatorFunctions::trans_product,
                    py::arg("a"), py::arg("b"), py::arg("trace_right"),
                    py::arg("noise") = 0.0);

    py::class_<OperatorTensor, shared_ptr<OperatorTensor>>(m, "OperatorTensor")
        .def(py::init<>())
        .def_readwrite("lmat", &OperatorTensor::lmat)
        .def_readwrite("rmat", &OperatorTensor::rmat)
        .def_readwrite("ops", &OperatorTensor::ops)
        .def("reallocate", &OperatorTensor::reallocate, py::arg("clean"))
        .def("deallocate", &OperatorTensor::deallocate);

    py::class_<DelayedOperatorTensor, shared_ptr<DelayedOperatorTensor>>(
        m, "DelayedOperatorTensor")
        .def(py::init<>())
        .def_readwrite("ops", &DelayedOperatorTensor::ops)
        .def_readwrite("mat", &DelayedOperatorTensor::mat)
        .def_readwrite("lops", &DelayedOperatorTensor::lops)
        .def_readwrite("rops", &DelayedOperatorTensor::rops)
        .def("reallocate", &DelayedOperatorTensor::reallocate, py::arg("clean"))
        .def("deallocate", &DelayedOperatorTensor::deallocate);

    py::bind_vector<vector<shared_ptr<OperatorTensor>>>(m, "VectorOpTensor");

    py::class_<TensorFunctions, shared_ptr<TensorFunctions>>(m,
                                                             "TensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions> &>())
        .def_readwrite("opf", &TensorFunctions::opf)
        .def_static("left_assign", &TensorFunctions::left_assign, py::arg("a"),
                    py::arg("c"))
        .def_static("right_assign", &TensorFunctions::right_assign,
                    py::arg("a"), py::arg("c"))
        .def("left_contract", &TensorFunctions::left_contract, py::arg("a"),
             py::arg("b"), py::arg("c"), py::arg("cexprs") = nullptr)
        .def("right_contract", &TensorFunctions::right_contract, py::arg("a"),
             py::arg("b"), py::arg("c"), py::arg("cexprs") = nullptr)
        .def("tensor_product_multiply",
             &TensorFunctions::tensor_product_multiply)
        .def("tensor_product_diagonal",
             &TensorFunctions::tensor_product_diagonal)
        .def("tensor_product", &TensorFunctions::tensor_product)
        .def("left_rotate", &TensorFunctions::left_rotate)
        .def("right_rotate", &TensorFunctions::right_rotate)
        .def("numerical_transform", &TensorFunctions::numerical_transform)
        .def("delayed_contract", (shared_ptr<DelayedOperatorTensor>(*)(
                                     const shared_ptr<OperatorTensor> &,
                                     const shared_ptr<OperatorTensor> &,
                                     const shared_ptr<OpExpr> &)) &
                                     TensorFunctions::delayed_contract)
        .def("delayed_contract_simplified",
             (shared_ptr<DelayedOperatorTensor>(*)(
                 const shared_ptr<OperatorTensor> &,
                 const shared_ptr<OperatorTensor> &,
                 const shared_ptr<Symbolic> &, const shared_ptr<Symbolic> &)) &
                 TensorFunctions::delayed_contract);

    py::class_<Partition, shared_ptr<Partition>>(m, "Partition")
        .def(py::init<const shared_ptr<OperatorTensor> &,
                      const shared_ptr<OperatorTensor> &,
                      const shared_ptr<OperatorTensor> &>())
        .def(py::init<const shared_ptr<OperatorTensor> &,
                      const shared_ptr<OperatorTensor> &,
                      const shared_ptr<OperatorTensor> &,
                      const shared_ptr<OperatorTensor> &>())
        .def_readwrite("left", &Partition::left)
        .def_readwrite("right", &Partition::right)
        .def_readwrite("middle", &Partition::middle)
        .def_readwrite("left_op_infos", &Partition::left_op_infos)
        .def_readwrite("right_op_infos", &Partition::right_op_infos)
        .def_static("find_op_info", &Partition::find_op_info)
        .def_static("build_left", &Partition::build_left)
        .def_static("build_right", &Partition::build_right)
        .def_static("get_uniq_labels", &Partition::get_uniq_labels)
        .def_static("get_uniq_sub_labels", &Partition::get_uniq_sub_labels)
        .def_static("deallocate_op_infos_notrunc",
                    &Partition::deallocate_op_infos_notrunc)
        .def_static("copy_op_infos", &Partition::copy_op_infos)
        .def_static("init_left_op_infos", &Partition::init_left_op_infos)
        .def_static("init_left_op_infos_notrunc",
                    &Partition::init_left_op_infos_notrunc)
        .def_static("init_right_op_infos", &Partition::init_right_op_infos)
        .def_static("init_right_op_infos_notrunc",
                    &Partition::init_right_op_infos_notrunc);

    py::bind_vector<vector<shared_ptr<Partition>>>(m, "VectorPartition");

    py::class_<EffectiveHamiltonian, shared_ptr<EffectiveHamiltonian>>(
        m, "EffectiveHamiltonian")
        .def(py::init<
             const vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>> &,
             const vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>> &,
             const shared_ptr<DelayedOperatorTensor> &,
             const shared_ptr<SparseMatrix> &, const shared_ptr<SparseMatrix> &,
             const shared_ptr<OpElement> &,
             const shared_ptr<SymbolicColumnVector> &,
             const shared_ptr<TensorFunctions> &, bool>())
        .def_readwrite("left_op_infos", &EffectiveHamiltonian::left_op_infos)
        .def_readwrite("right_op_infos", &EffectiveHamiltonian::right_op_infos)
        .def_readwrite("op", &EffectiveHamiltonian::op)
        .def_readwrite("bra", &EffectiveHamiltonian::bra)
        .def_readwrite("ket", &EffectiveHamiltonian::ket)
        .def_readwrite("diag", &EffectiveHamiltonian::diag)
        .def_readwrite("cmat", &EffectiveHamiltonian::cmat)
        .def_readwrite("vmat", &EffectiveHamiltonian::vmat)
        .def_readwrite("tf", &EffectiveHamiltonian::tf)
        .def_readwrite("opdq", &EffectiveHamiltonian::opdq)
        .def_readwrite("compute_diag", &EffectiveHamiltonian::compute_diag)
        .def("__call__", &EffectiveHamiltonian::operator(), py::arg("b"),
             py::arg("c"), py::arg("idx") = 0, py::arg("factor") = 1.0)
        .def("eigs", &EffectiveHamiltonian::eigs)
        .def("multiply", &EffectiveHamiltonian::multiply)
        .def("expect", &EffectiveHamiltonian::expect)
        .def("rk4_apply", &EffectiveHamiltonian::rk4_apply, py::arg("beta"),
             py::arg("const_e"), py::arg("eval_energy") = false)
        .def("expo_apply", &EffectiveHamiltonian::expo_apply, py::arg("beta"),
             py::arg("const_e"), py::arg("iprint") = false)
        .def("deallocate", &EffectiveHamiltonian::deallocate);

    py::enum_<FuseTypes>(m, "FuseTypes", py::arithmetic())
        .value("NoFuse", FuseTypes::NoFuse)
        .value("FuseL", FuseTypes::FuseL)
        .value("FuseR", FuseTypes::FuseR)
        .value("FuseLR", FuseTypes::FuseLR);

    py::class_<MovingEnvironment, shared_ptr<MovingEnvironment>>(
        m, "MovingEnvironment")
        .def(py::init<const shared_ptr<MPO> &, const shared_ptr<MPS> &,
                      const shared_ptr<MPS> &>())
        .def(py::init<const shared_ptr<MPO> &, const shared_ptr<MPS> &,
                      const shared_ptr<MPS> &, const string &>())
        .def_readwrite("n_sites", &MovingEnvironment::n_sites)
        .def_readwrite("center", &MovingEnvironment::center)
        .def_readwrite("dot", &MovingEnvironment::dot)
        .def_readwrite("mpo", &MovingEnvironment::mpo)
        .def_readwrite("bra", &MovingEnvironment::bra)
        .def_readwrite("ket", &MovingEnvironment::ket)
        .def_readwrite("envs", &MovingEnvironment::envs)
        .def_readwrite("tag", &MovingEnvironment::tag)
        .def("left_contract_rotate", &MovingEnvironment::left_contract_rotate)
        .def("right_contract_rotate", &MovingEnvironment::right_contract_rotate)
        .def("init_environments", &MovingEnvironment::init_environments)
        .def("prepare", &MovingEnvironment::prepare)
        .def("move_to", &MovingEnvironment::move_to)
        .def("eff_ham", &MovingEnvironment::eff_ham, py::arg("fuse_type"),
             py::arg("compute_diag"))
        .def("get_left_partition_filename",
             &MovingEnvironment::get_left_partition_filename)
        .def("get_right_partition_filename",
             &MovingEnvironment::get_right_partition_filename)
        .def_static("contract_two_dot", &MovingEnvironment::contract_two_dot,
                    py::arg("i"), py::arg("mps"), py::arg("reduced") = false)
        .def_static("density_matrix", &MovingEnvironment::density_matrix,
                    py::arg("opdq"), py::arg("psi"), py::arg("trace_right"),
                    py::arg("noise"))
        .def_static("density_matrix_with_weights",
                    &MovingEnvironment::density_matrix_with_weights,
                    py::arg("opdq"), py::arg("psi"), py::arg("trace_right"),
                    py::arg("noise"), py::arg("mats"), py::arg("weights"))
        .def_static(
            "split_density_matrix",
            [](const shared_ptr<SparseMatrix> &dm,
               const shared_ptr<SparseMatrix> &wfn, int k, bool trace_right) {
                shared_ptr<SparseMatrix> left = nullptr, right = nullptr;
                double error = MovingEnvironment::split_density_matrix(
                    dm, wfn, k, trace_right, left, right);
                return make_tuple(error, left, right);
            })
        .def_static("propagate_wfn", &MovingEnvironment::propagate_wfn,
                    py::arg("i"), py::arg("n_sites"), py::arg("mps"),
                    py::arg("forward"));

    py::class_<Hamiltonian, shared_ptr<Hamiltonian>>(m, "Hamiltonian")
        .def(py::init<SpinLabel, SpinLabel, int, bool,
                      const shared_ptr<FCIDUMP> &, const vector<uint8_t> &>())
        .def_static("swap_d2h", &Hamiltonian::swap_d2h)
        .def_readwrite("n_syms", &Hamiltonian::n_syms)
        .def_readwrite("opf", &Hamiltonian::opf)
        .def_readwrite("mu", &Hamiltonian::mu)
        .def_readwrite("n_sites", &Hamiltonian::n_sites)
        .def_readwrite("orb_sym", &Hamiltonian::orb_sym)
        .def_readwrite("vaccum", &Hamiltonian::vaccum)
        .def_readwrite("target", &Hamiltonian::target)
        .def_readwrite("su2", &Hamiltonian::su2)
        .def_property_readonly("basis",
                               [](Hamiltonian *self) {
                                   return Array<StateInfo>(self->basis,
                                                           self->n_syms);
                               })
        .def_property_readonly("op_prims",
                               [](Hamiltonian *self) {
                                   return make_pair(self->op_prims[0],
                                                    self->op_prims[1]);
                               })
        .def("v", &Hamiltonian::v)
        .def("t", &Hamiltonian::t)
        .def("e", &Hamiltonian::e)
        .def_property_readonly(
            "site_op_infos",
            [](Hamiltonian *self) {
                return Array<
                    vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>>>(
                    self->site_op_infos, self->n_syms);
            })
        .def("init_site_ops", &Hamiltonian::init_site_ops)
        .def("get_site_ops", &Hamiltonian::get_site_ops)
        .def("filter_site_ops", &Hamiltonian::filter_site_ops)
        .def("find_site_op_info", &Hamiltonian::find_site_op_info)
        .def("find_site_norm_op", &Hamiltonian::find_site_norm_op)
        .def("deallocate", &Hamiltonian::deallocate);

    py::class_<DMRG::Iteration, shared_ptr<DMRG::Iteration>>(m, "DMRGIteration")
        .def(py::init<double, double, int, size_t, double>())
        .def(py::init<double, double, int>())
        .def_readwrite("energy", &DMRG::Iteration::energy)
        .def_readwrite("error", &DMRG::Iteration::error)
        .def_readwrite("ndav", &DMRG::Iteration::ndav)
        .def_readwrite("tdav", &DMRG::Iteration::tdav)
        .def_readwrite("nflop", &DMRG::Iteration::nflop)
        .def("__repr__", [](DMRG::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<DMRG, shared_ptr<DMRG>>(m, "DMRG")
        .def(py::init<const shared_ptr<MovingEnvironment> &,
                      const vector<uint16_t> &, const vector<double> &>())
        .def_readwrite("me", &DMRG::me)
        .def_readwrite("bond_dims", &DMRG::bond_dims)
        .def_readwrite("noises", &DMRG::noises)
        .def_readwrite("energies", &DMRG::energies)
        .def_readwrite("forward", &DMRG::forward)
        .def("update_two_dot", &DMRG::update_two_dot)
        .def("blocking", &DMRG::blocking)
        .def("sweep", &DMRG::sweep)
        .def("solve", &DMRG::solve, py::arg("n_sweeps"),
             py::arg("forward") = true, py::arg("tol") = 1E-6);

    py::enum_<TETypes>(m, "TETypes", py::arithmetic())
        .value("TangentSpace", TETypes::TangentSpace)
        .value("RK4", TETypes::RK4);

    py::class_<ImaginaryTE::Iteration, shared_ptr<ImaginaryTE::Iteration>>(
        m, "ImaginaryTEIteration")
        .def(py::init<double, double, double, int, int, size_t, double>())
        .def(py::init<double, double, double, int, int>())
        .def_readwrite("energy", &ImaginaryTE::Iteration::energy)
        .def_readwrite("normsq", &ImaginaryTE::Iteration::normsq)
        .def_readwrite("error", &ImaginaryTE::Iteration::error)
        .def_readwrite("nexpo", &ImaginaryTE::Iteration::nexpo)
        .def_readwrite("nexpok", &ImaginaryTE::Iteration::nexpok)
        .def_readwrite("texpo", &ImaginaryTE::Iteration::texpo)
        .def_readwrite("nflop", &ImaginaryTE::Iteration::nflop)
        .def("__repr__", [](ImaginaryTE::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<ImaginaryTE, shared_ptr<ImaginaryTE>>(m, "ImaginaryTE")
        .def(py::init<const shared_ptr<MovingEnvironment> &,
                      const vector<uint16_t> &, TETypes>())
        .def_readwrite("me", &ImaginaryTE::me)
        .def_readwrite("bond_dims", &ImaginaryTE::bond_dims)
        .def_readwrite("noises", &ImaginaryTE::noises)
        .def_readwrite("energies", &ImaginaryTE::energies)
        .def_readwrite("normsqs", &ImaginaryTE::normsqs)
        .def_readwrite("forward", &ImaginaryTE::forward)
        .def_readwrite("mode", &ImaginaryTE::mode)
        .def("update_two_dot", &ImaginaryTE::update_two_dot)
        .def("blocking", &ImaginaryTE::blocking)
        .def("sweep", &ImaginaryTE::sweep)
        .def("normalize", &ImaginaryTE::normalize)
        .def("solve", &ImaginaryTE::solve, py::arg("n_sweeps"), py::arg("beta"),
             py::arg("forward") = true, py::arg("tol") = 1E-6);

    py::class_<Compress::Iteration, shared_ptr<Compress::Iteration>>(
        m, "CompressIteration")
        .def(py::init<double, double, size_t, double>())
        .def(py::init<double, double>())
        .def_readwrite("norm", &Compress::Iteration::norm)
        .def_readwrite("error", &Compress::Iteration::error)
        .def_readwrite("tmult", &Compress::Iteration::tmult)
        .def_readwrite("nflop", &Compress::Iteration::nflop)
        .def("__repr__", [](Compress::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<Compress, shared_ptr<Compress>>(m, "Compress")
        .def(py::init<const shared_ptr<MovingEnvironment> &,
                      const vector<uint16_t> &, const vector<uint16_t> &,
                      const vector<double> &>())
        .def_readwrite("me", &Compress::me)
        .def_readwrite("bra_bond_dims", &Compress::bra_bond_dims)
        .def_readwrite("ket_bond_dims", &Compress::ket_bond_dims)
        .def_readwrite("noises", &Compress::noises)
        .def_readwrite("norms", &Compress::norms)
        .def_readwrite("forward", &Compress::forward)
        .def("update_two_dot", &Compress::update_two_dot)
        .def("blocking", &Compress::blocking)
        .def("sweep", &Compress::sweep)
        .def("solve", &Compress::solve, py::arg("n_sweeps"),
             py::arg("forward") = true, py::arg("tol") = 1E-6);

    py::class_<Expect::Iteration, shared_ptr<Expect::Iteration>>(
        m, "ExpectIteration")
        .def(py::init<const vector<pair<shared_ptr<OpExpr>, double>> &, double,
                      double, size_t, double>())
        .def(py::init<const vector<pair<shared_ptr<OpExpr>, double>> &, double,
                      double>())
        .def_readwrite("bra_error", &Expect::Iteration::bra_error)
        .def_readwrite("ket_error", &Expect::Iteration::ket_error)
        .def_readwrite("tmult", &Expect::Iteration::tmult)
        .def_readwrite("nflop", &Expect::Iteration::nflop)
        .def("__repr__", [](Expect::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<Expect, shared_ptr<Expect>>(m, "Expect")
        .def(py::init<const shared_ptr<MovingEnvironment> &, uint16_t,
                      uint16_t>())
        .def_readwrite("me", &Expect::me)
        .def_readwrite("bra_bond_dim", &Expect::bra_bond_dim)
        .def_readwrite("ket_bond_dim", &Expect::ket_bond_dim)
        .def_readwrite("expectations", &Expect::expectations)
        .def_readwrite("forward", &Expect::forward)
        .def("update_two_dot", &Expect::update_two_dot)
        .def("blocking", &Expect::blocking)
        .def("sweep", &Expect::sweep)
        .def("solve", &Expect::solve, py::arg("propagate"),
             py::arg("forward") = true)
        .def("get_1pdm_spatial", &Expect::get_1pdm_spatial,
             py::arg("n_physical_sites") = (uint16_t)0U);

    py::class_<MPOSchemer, shared_ptr<MPOSchemer>>(m, "MPOSchemer")
        .def_readwrite("left_trans_site", &MPOSchemer::left_trans_site)
        .def_readwrite("right_trans_site", &MPOSchemer::right_trans_site)
        .def_readwrite("left_new_operator_names",
                       &MPOSchemer::left_new_operator_names)
        .def_readwrite("right_new_operator_names",
                       &MPOSchemer::right_new_operator_names)
        .def_readwrite("left_new_operator_exprs",
                       &MPOSchemer::left_new_operator_exprs)
        .def_readwrite("right_new_operator_exprs",
                       &MPOSchemer::right_new_operator_exprs)
        .def(py::init<uint8_t, uint8_t>())
        .def("copy", &MPOSchemer::copy)
        .def("get_transform_formulas", &MPOSchemer::get_transform_formulas);

    py::class_<MPO, shared_ptr<MPO>>(m, "MPO")
        .def(py::init<int>())
        .def_readwrite("n_sites", &MPO::n_sites)
        .def_readwrite("const_e", &MPO::const_e)
        .def_readwrite("tensors", &MPO::tensors)
        .def_readwrite("left_operator_names", &MPO::left_operator_names)
        .def_readwrite("right_operator_names", &MPO::right_operator_names)
        .def_readwrite("middle_operator_names", &MPO::middle_operator_names)
        .def_readwrite("left_operator_exprs", &MPO::left_operator_exprs)
        .def_readwrite("right_operator_exprs", &MPO::right_operator_exprs)
        .def_readwrite("middle_operator_exprs", &MPO::middle_operator_exprs)
        .def_readwrite("op", &MPO::op)
        .def_readwrite("schemer", &MPO::schemer)
        .def_readwrite("tf", &MPO::tf)
        .def_readwrite("site_op_infos", &MPO::site_op_infos)
        .def_readwrite("schemer", &MPO::schemer)
        .def("get_blocking_formulas", &MPO::get_blocking_formulas)
        .def("get_ancilla_type", &MPO::get_ancilla_type)
        .def("deallocate", &MPO::deallocate);

    py::class_<Rule, shared_ptr<Rule>>(m, "Rule")
        .def(py::init<>())
        .def("__call__", &Rule::operator());

    py::class_<NoTransposeRule, shared_ptr<NoTransposeRule>, Rule>(
        m, "NoTransposeRule")
        .def_readwrite("prim_rule", &NoTransposeRule::prim_rule)
        .def(py::init<const shared_ptr<Rule> &>());

    py::class_<RuleQCSU2, shared_ptr<RuleQCSU2>, Rule>(m, "RuleQCSU2")
        .def(py::init<>())
        .def(py::init<bool, bool, bool, bool, bool, bool>());

    py::enum_<QCTypes>(m, "QCTypes", py::arithmetic())
        .value("NC", QCTypes::NC)
        .value("CN", QCTypes::CN)
        .value("NCCN", QCTypes(QCTypes::NC | QCTypes::CN))
        .value("Conventional", QCTypes::Conventional);

    py::class_<SimplifiedMPO, shared_ptr<SimplifiedMPO>, MPO>(m,
                                                              "SimplifiedMPO")
        .def_readwrite("prim_mpo", &SimplifiedMPO::prim_mpo)
        .def_readwrite("rule", &SimplifiedMPO::rule)
        .def_readwrite("collect_terms", &SimplifiedMPO::collect_terms)
        .def(py::init<const shared_ptr<MPO> &, const shared_ptr<Rule> &>())
        .def(
            py::init<const shared_ptr<MPO> &, const shared_ptr<Rule> &, bool>())
        .def("simplify_expr", &SimplifiedMPO::simplify_expr)
        .def("simplify_symbolic", &SimplifiedMPO::simplify_symbolic)
        .def("simplify", &SimplifiedMPO::simplify);

    py::class_<IdentityMPO, shared_ptr<IdentityMPO>, MPO>(m, "IdentityMPO")
        .def(py::init<const Hamiltonian &>());

    py::class_<MPOQCSU2, shared_ptr<MPOQCSU2>, MPO>(m, "MPOQCSU2")
        .def_readwrite("mode", &MPOQCSU2::mode)
        .def(py::init<const Hamiltonian &>())
        .def(py::init<const Hamiltonian &, QCTypes>());

    py::class_<PDM1MPOQCSU2, shared_ptr<PDM1MPOQCSU2>, MPO>(m, "PDM1MPOQCSU2")
        .def(py::init<const Hamiltonian &>());

    py::class_<AncillaMPO, shared_ptr<AncillaMPO>, MPO>(m, "AncillaMPO")
        .def_readwrite("n_physical_sites", &AncillaMPO::n_physical_sites)
        .def_readwrite("prim_mpo", &AncillaMPO::prim_mpo)
        .def(py::init<const shared_ptr<MPO> &>())
        .def(py::init<const shared_ptr<MPO> &, bool>());
}
