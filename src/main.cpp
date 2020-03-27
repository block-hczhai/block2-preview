
#include "quantum.hpp"

using namespace std;
using namespace block2;

// block2 d2h numbers:
// 0,  1,   2,   3,   4,   5,   6,   7
// Ag, B3u, B2u, B1g, B1u, B2g, B3g, Au
//
// CheMPS2 d2h numbers:
// 0  1   2   3   4  5   6   7
// Ag B1g B2g B3g Au B1u B2u B3u

int swap_d2h[] = { -1, 0, 7, 6, 1, 5, 2, 3, 4 };
int cr2_orbsym[] = { 1,1,1,1,1,1,1,1,1,1,1,5,5,5,5,5,5,5,5,5,5,5,2,2,2,2,6,6,6,6,3,3,3,3,7,7,7,7,4,4,8,8 };
int c2_orbsym[] = { 1,1,1,1,1,1,2,2,2,3,3,3,4,5,5,5,5,5,5,6,6,6,7,7,7,8 };
int n2_orbsym[] = { 1,1,1,6,7,5,5,5,3,2 };

int n_elec = 48, n_orb = 42;

size_t isize = 1E7;
size_t dsize = 1E7;

// CR2:
// ORBSYM=1,5,1,5,1,5,2,3,6,7,1,5,1,2,3,6,7,5,1,2,3,1,4,1,5,8,5,6,7,5,1,5,6,7,1,4,5,3,2,1,8,5

int main(int argc, char *argv[]) {
    
    // allocator
    ialloc = new StackAllocator<uint32_t>(new uint32_t[isize], isize);
    dalloc = new StackAllocator<double>(new double[dsize], dsize);
    
    // system info
    SpinLabel vaccum;
    SpinLabel target(n_elec, 0, 0);
    int *orb_sym = new int[n_orb];
    for (int i = 0; i < n_orb; i++)
        orb_sym[i] = swap_d2h[cr2_orbsym[i]];
    
    StateInfo *basis = new StateInfo[n_orb];
    StateInfo *left_dims_fci = new StateInfo[n_orb + 1];
    StateInfo *right_dims_fci = new StateInfo[n_orb + 1];
    
    // basis states
    for (int i = 0; i < n_orb; i++) {
        basis[i].allocate(3);
        basis[i].quanta[0] = vaccum;
        basis[i].quanta[2] = SpinLabel(1, 1, orb_sym[i]);
        basis[i].quanta[1] = SpinLabel(2, 0, 0);
        basis[i].n_states[0] = basis[i].n_states[1] = basis[i].n_states[2] = 1;
        basis[i].sort_states();
    }
    
    // left dims
    left_dims_fci[0] = StateInfo(vaccum);
    cout << left_dims_fci[0] << endl;
    cout << *ialloc << endl;
    for (int i = 0; i < n_orb; i++) {
        left_dims_fci[i + 1] = StateInfo::tensor_product(left_dims_fci[i], basis[i], target);
        cout << i << " " << left_dims_fci[i + 1].n_states_total << endl;
    }
    
    cout << *ialloc << endl;
    
    // right dims
    right_dims_fci[n_orb] = StateInfo(vaccum);
    cout << right_dims_fci[n_orb] << endl;
    cout << *ialloc << endl;
    for (int i = n_orb - 1; i >= 0; i--) {
        right_dims_fci[i] = StateInfo::tensor_product(basis[i], right_dims_fci[i + 1], target);
        cout << i << " " << right_dims_fci[i].n_states_total << endl;
    }
    
    cout << *ialloc << endl;
    
    // filter
    for (int i = 0; i <= n_orb; i++) {
        StateInfo::filter(left_dims_fci[i], right_dims_fci[i], target);
        cout << right_dims_fci[i] << endl;
    }
    
    for (int i = 0; i <= n_orb; i++) {
        left_dims_fci[i].collect();
        cout << i << " " << left_dims_fci[i].n << endl;
    }
    
    for (int i = n_orb; i >= 0; i--) {
        right_dims_fci[i].collect();
        cout << i << " " << right_dims_fci[i].n << endl;
    }
    
    cout << *ialloc << endl;
    for (int i = 0; i <= n_orb; i++)
        right_dims_fci[i].deallocate();
    cout << *ialloc << endl;
    for (int i = n_orb; i >= 0; i--)
        left_dims_fci[i].deallocate();
    cout << *ialloc << endl;
    return 0;
}