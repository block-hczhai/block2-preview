
#include "quantum.hpp"
#include <tuple>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

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
        [](size_t isize, size_t dsize) {
            ialloc = new StackAllocator<uint32_t>(new uint32_t[isize], isize);
            dalloc = new StackAllocator<double>(new double[dsize], dsize);
        },
        py::arg("isize") = size_t(1E7), py::arg("dsize") = size_t(5E8));

    m.def("release_memory", []() {
        assert(ialloc->used == 0 && dalloc->used == 0);
        delete[] ialloc->data;
        delete[] dalloc->data;
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

    py::class_<Global>(m, "Global")
        .def_property_static(
            "ialloc", [](py::object) { return ialloc; },
            [](py::object, StackAllocator<uint32_t> *ia) { ialloc = ia; })
        .def_property_static(
            "dalloc", [](py::object) { return dalloc; },
            [](py::object, StackAllocator<double> *da) { dalloc = da; });

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

    py::class_<Random>(m, "Random")
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

    py::class_<FCIDUMP, shared_ptr<FCIDUMP>>(m, "FCIDUMP")
        .def(py::init<>())
        .def("read", &FCIDUMP::read)
        .def("deallocate", &FCIDUMP::deallocate)
        .def_property_readonly("orb_sym", &FCIDUMP::orb_sym)
        .def_property_readonly("n_elec", &FCIDUMP::n_elec)
        .def_property_readonly("twos", &FCIDUMP::twos)
        .def_property_readonly("isym", &FCIDUMP::isym)
        .def_property_readonly("n_sites", &FCIDUMP::n_sites)
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
        .value("PDM1", OpNames::PDM1);

    py::enum_<OpTypes>(m, "OpTypes", py::arithmetic())
        .value("Zero", OpTypes::Zero)
        .value("Elem", OpTypes::Elem)
        .value("Prod", OpTypes::Prod)
        .value("Sum", OpTypes::Sum);

    py::class_<OpExpr, shared_ptr<OpExpr>>(m, "OpExpr")
        .def(py::init<>())
        .def("get_type", &OpExpr::get_type)
        .def(py::self == py::self)
        .def("__repr__", &to_str);

    py::class_<OpElement, shared_ptr<OpElement>, OpExpr>(m, "OpElement")
        .def(py::init<OpNames, const vector<uint8_t> &, SpinLabel>())
        .def(py::init<OpNames, const vector<uint8_t> &, SpinLabel, double>())
        .def_readwrite("name", &OpElement::name)
        .def_readwrite("site_index", &OpElement::site_index)
        .def_readwrite("factor", &OpElement::factor)
        .def_readwrite("q_label", &OpElement::q_label)
        .def("abs", &OpElement::abs)
        .def("__mul__", &OpElement::operator*)
        .def(py::self == py::self)
        .def(py::self < py::self)
        .def("__hash__", &OpElement::hash);

    py::class_<OpString, shared_ptr<OpString>, OpExpr>(m, "OpString")
        .def(py::init<const vector<shared_ptr<OpElement>> &, double>())
        .def_readwrite("factor", &OpString::factor)
        .def_readwrite("ops", &OpString::ops);

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
        .def_property_readonly("n_states", [](StateInfo *self) {
            return Array<uint16_t>(self->n_states, self->n);
        });

    py::class_<SparseMatrixInfo::CollectedInfo,
               shared_ptr<SparseMatrixInfo::CollectedInfo>>(m, "CollectedInfo")
        .def(py::init<>())
        .def_readwrite("n", &SparseMatrixInfo::CollectedInfo::n)
        .def_readwrite("nc", &SparseMatrixInfo::CollectedInfo::nc)
        .def("__repr__", [](SparseMatrixInfo::CollectedInfo *self) {
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
        .def_readwrite("conj", &SparseMatrix::conj)
        .def_readwrite("total_memory", &SparseMatrix::total_memory)
        .def_property_readonly("data",
                               [](SparseMatrix *self) {
                                   return py::array_t<double>(
                                       self->total_memory, self->data);
                               })
        .def("clear", &SparseMatrix::clear)
        .def("copy_data", &SparseMatrix::copy_data)
        .def("allocate",
             [](SparseMatrix *self, const shared_ptr<SparseMatrixInfo> &info) {
                 self->allocate(info);
             })
        .def("deallocate", &SparseMatrix::deallocate)
        .def("reallocate", &SparseMatrix::reallocate, py::arg("length"))
        .def("__getitem__",
             [](SparseMatrix *self, int idx) { return (*self)[idx]; })
        .def("left_canonicalize", &SparseMatrix::left_canonicalize,
             py::arg("rmat"))
        .def("right_canonicalize", &SparseMatrix::right_canonicalize,
             py::arg("lmat"))
        .def("randomize", &SparseMatrix::randomize)
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

    py::class_<MPSInfo, shared_ptr<MPSInfo>>(m, "MPSInfo")
        .def(py::init([](int n_sites, SpinLabel vaccum, SpinLabel target,
                         Array<StateInfo> &basis, py::array_t<uint8_t> &orbsym,
                         uint8_t n_syms) {
            return make_shared<MPSInfo>(n_sites, vaccum, target, basis.data,
                                        orbsym.mutable_data(), n_syms);
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
        .def("set_bond_dimension", &MPSInfo::set_bond_dimension)
        .def("deallocate", &MPSInfo::deallocate);

    py::class_<MPS, shared_ptr<MPS>>(m, "MPS")
        .def(py::init<int, int, int>())
        .def_readwrite("n_sites", &MPS::n_sites)
        .def_readwrite("center", &MPS::center)
        .def_readwrite("dot", &MPS::dot)
        .def_readwrite("info", &MPS::info)
        .def_readwrite("mat_infos", &MPS::mat_infos)
        .def_readwrite("tensors", &MPS::tensors)
        .def_readwrite("canonical_form", &MPS::canonical_form)
        .def("initialize", &MPS::initialize)
        .def("random_canonicalize", &MPS::random_canonicalize)
        .def("deallocate", &MPS::deallocate);

    py::class_<CG, shared_ptr<CG>>(m, "CG").def(py::init<>());

    py::class_<OperatorFunctions, shared_ptr<OperatorFunctions>>(
        m, "OperatorFunctions")
        .def(py::init<const shared_ptr<CG> &>())
        .def("tensor_product", &OperatorFunctions::tensor_product);

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
             py::arg("b"), py::arg("c"))
        .def("right_contract", &TensorFunctions::right_contract, py::arg("a"),
             py::arg("b"), py::arg("c"));

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
        .def_readwrite("left_op_infos_notrunc",
                       &Partition::left_op_infos_notrunc)
        .def_readwrite("right_op_infos", &Partition::right_op_infos)
        .def_readwrite("right_op_infos_notrunc",
                       &Partition::right_op_infos_notrunc)
        .def("deallocate_left_op_infos", &Partition::deallocate_left_op_infos)
        .def("deallocate_right_op_infos",
             &Partition::deallocate_right_op_infos);

    py::bind_vector<vector<shared_ptr<Partition>>>(m, "VectorPartition");

    py::class_<EffectiveHamiltonian, shared_ptr<EffectiveHamiltonian>>(
        m, "EffectiveHamiltonian")
        .def(py::init<
             const vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>> &,
             const vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>> &,
             const shared_ptr<DelayedOperatorTensor> &,
             const shared_ptr<SparseMatrix> &, const shared_ptr<OpElement> &,
             const shared_ptr<SymbolicColumnVector> &,
             const shared_ptr<TensorFunctions> &>())
        .def_readwrite("left_op_infos_notrunc",
                       &EffectiveHamiltonian::left_op_infos_notrunc)
        .def_readwrite("right_op_infos_notrunc",
                       &EffectiveHamiltonian::right_op_infos_notrunc)
        .def_readwrite("op", &EffectiveHamiltonian::op)
        .def_readwrite("psi", &EffectiveHamiltonian::psi)
        .def_readwrite("diag", &EffectiveHamiltonian::diag)
        .def_readwrite("cmat", &EffectiveHamiltonian::cmat)
        .def_readwrite("vmat", &EffectiveHamiltonian::vmat)
        .def_readwrite("tf", &EffectiveHamiltonian::tf)
        .def_readwrite("opdq", &EffectiveHamiltonian::opdq)
        .def("eigs", &EffectiveHamiltonian::eigs)
        .def("deallocate", &EffectiveHamiltonian::deallocate);

    py::class_<MovingEnvironment, shared_ptr<MovingEnvironment>>(
        m, "MovingEnvironment")
        .def(py::init(
            [](const shared_ptr<MPO> &mpo, const shared_ptr<MPS> &bra,
               const shared_ptr<MPS> &ket,
               const shared_ptr<TensorFunctions> &tf,
               Array<vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>>>
                   &site_op_infos) {
                return make_shared<MovingEnvironment>(mpo, bra, ket, tf,
                                                      site_op_infos.data);
            }))
        .def_readwrite("n_sites", &MovingEnvironment::n_sites)
        .def_readwrite("center", &MovingEnvironment::center)
        .def_readwrite("dot", &MovingEnvironment::dot)
        .def_readwrite("mpo", &MovingEnvironment::mpo)
        .def_readwrite("bra", &MovingEnvironment::bra)
        .def_readwrite("ket", &MovingEnvironment::ket)
        .def_readwrite("envs", &MovingEnvironment::envs)
        .def_readwrite("tf", &MovingEnvironment::tf)
        .def_property_readonly(
            "site_op_infos",
            [](MovingEnvironment *self) {
                return Array<
                    vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>>>(
                    self->site_op_infos, self->ket->info->n_syms);
            })
        .def("init_environments", &MovingEnvironment::init_environments)
        .def("prepare", &MovingEnvironment::prepare)
        .def("move_to", &MovingEnvironment::move_to)
        .def("eff_ham", &MovingEnvironment::eff_ham)
        .def("density_matrix", &MovingEnvironment::density_matrix)
        .def("split_density_matrix",
             [](MovingEnvironment *self, const shared_ptr<SparseMatrix> &dm,
                const shared_ptr<SparseMatrix> &wfn, int k, bool trace_right) {
                 shared_ptr<SparseMatrix> left = nullptr, right = nullptr;
                 double error = self->split_density_matrix(
                     dm, wfn, k, trace_right, left, right);
                 return make_tuple(error, left, right);
             })
        .def("deallocate", &MovingEnvironment::deallocate);

    py::class_<Hamiltonian, shared_ptr<Hamiltonian>>(m, "Hamiltonian")
        .def(py::init<SpinLabel, SpinLabel, int, bool,
                      const shared_ptr<FCIDUMP> &, const vector<uint8_t> &>())
        .def_static("swap_d2h", &Hamiltonian::swap_d2h)
        .def_readwrite("n_syms", &Hamiltonian::n_syms)
        .def_readwrite("opf", &Hamiltonian::opf)
        .def_property_readonly("basis",
                               [](Hamiltonian *self) {
                                   return Array<StateInfo>(self->basis,
                                                           self->n_syms);
                               })
        .def_property_readonly("orb_sym",
                               [](Hamiltonian *self) {
                                   return py::array_t<uint8_t>(
                                       self->orb_sym.size(), &self->orb_sym[0]);
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
        .def("deallocate", &Hamiltonian::deallocate);
    
    py::class_<DMRG::Iteration, shared_ptr<DMRG::Iteration>>(m, "Iteration")
        .def(py::init<double, double, int>())
        .def_readwrite("energy", &DMRG::Iteration::energy)
        .def_readwrite("error", &DMRG::Iteration::error)
        .def_readwrite("ndav", &DMRG::Iteration::ndav)
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
        .def("contract_two_dot", &DMRG::contract_two_dot)
        .def("update_two_dot", &DMRG::update_two_dot)
        .def("blocking", &DMRG::blocking)
        .def("sweep", &DMRG::sweep)
        .def("solve", &DMRG::solve, py::arg("n_sweeps"), py::arg("tol") = 1E-6,
             py::arg("forward") = true);

    py::class_<MPO, shared_ptr<MPO>>(m, "MPO")
        .def(py::init<int>())
        .def_readwrite("n_sites", &MPO::n_sites)
        .def_readwrite("tensors", &MPO::tensors)
        .def_readwrite("left_operator_names", &MPO::left_operator_names)
        .def_readwrite("right_operator_names", &MPO::right_operator_names)
        .def_readwrite("middle_operator_names", &MPO::middle_operator_names)
        .def("deallocate", &MPO::deallocate);

    py::class_<QCMPO, shared_ptr<QCMPO>, MPO>(m, "QCMPO")
        .def(py::init<const Hamiltonian &>())
        .def("deallocate", &QCMPO::deallocate);
}
