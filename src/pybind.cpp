
#include "quantum.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace block2;

PYBIND11_MAKE_OPAQUE(vector<int>);
PYBIND11_MAKE_OPAQUE(vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>>);

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
        .def_static("rand_seed", &Random::rand_seed, py::arg("i") = 0U);

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
        .def(py::self - py::self);

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

    py::class_<SparseMatrixInfo, shared_ptr<SparseMatrixInfo>>(
        m, "SparseMatrixInfo");

    py::bind_vector<vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>>>(
        m, "VectorPLMatInfo");

    py::class_<MPSInfo, shared_ptr<MPSInfo>>(m, "MPSInfo")
        .def(py::init([](int n_sites, SpinLabel vaccum, SpinLabel target,
                         Array<StateInfo> &basis, Array<uint8_t> &orbsym,
                         uint8_t n_syms) {
            return make_shared<MPSInfo>(n_sites, vaccum, target, basis.data,
                                        orbsym.data, n_syms);
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
        .def_readwrite("canonical_form", &MPS::canonical_form)
        .def("initialize", &MPS::initialize)
        .def("random_canonicalize", &MPS::random_canonicalize)
        .def("deallocate", &MPS::deallocate);

    py::class_<CG, shared_ptr<CG>>(m, "CG").def(py::init<>());

    py::class_<OperatorFunctions, shared_ptr<OperatorFunctions>>(
        m, "OperatorFunctions")
        .def(py::init<const shared_ptr<CG> &>());

    py::class_<TensorFunctions, shared_ptr<TensorFunctions>>(m,
                                                             "TensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions> &>());

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
        .def("init_environments", &MovingEnvironment::init_environments)
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
                                   return Array<uint8_t>(&self->orb_sym[0],
                                                         self->orb_sym.size());
                               })
        .def_property_readonly(
            "site_op_infos",
            [](Hamiltonian *self) {
                return Array<
                    vector<pair<SpinLabel, shared_ptr<SparseMatrixInfo>>>>(
                    self->site_op_infos, self->n_syms);
            })
        .def("deallocate", &Hamiltonian::deallocate);

    py::class_<MPO, shared_ptr<MPO>>(m, "MPO")
        .def(py::init<int>())
        .def("deallocate", &MPO::deallocate);

    py::class_<QCMPO, shared_ptr<QCMPO>, MPO>(m, "QCMPO")
        .def(py::init<const Hamiltonian &>())
        .def("deallocate", &QCMPO::deallocate);
}