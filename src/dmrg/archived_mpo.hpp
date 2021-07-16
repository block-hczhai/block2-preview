
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

/** Store tensors in MPO to save memory cost. */

#pragma once

#include "../core/archived_tensor_functions.hpp"
#include "mpo.hpp"
#include <memory>

using namespace std;

namespace block2 {

/** An MPO with site operator stored in disk.
 * Note that this allows only loading one site operator at a time, which can
 * greatly save memory. But the IO overhead can be extremely high. The better
 * solution should be using ``archive_filename`` in ``MPO``, which loads all
 * data in a site tensor at a time.
 * @tparam S Quantum label type.
 */
template <typename S> struct ArchivedMPO : MPO<S> {
    using MPO<S>::n_sites;
    /** Constructor.
     * @param mpo The original MPO.
     * @param tag The tag for constructing a unique filename for this MPO.
     */
    ArchivedMPO(const shared_ptr<MPO<S>> &mpo, const string &tag = "MPO")
        : MPO<S>(*mpo) {
        shared_ptr<ArchivedTensorFunctions<S>> artf =
            make_shared<ArchivedTensorFunctions<S>>(mpo->tf->opf);
        MPO<S>::tf = artf;
        artf->filename =
            frame->save_dir + "/" + frame->prefix_distri + ".AR." + tag;
        artf->offset = 0;
        for (int16_t m = n_sites - 1; m >= 0; m--)
            artf->archive_tensor(MPO<S>::tensors[m]);
    }
    /** Deallocate operator data in this MPO.
     * Since all MPO tensor data is stored in disk, this method does nothing.
     */
    void deallocate() override {}
};

} // namespace block2
