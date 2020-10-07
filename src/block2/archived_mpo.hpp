
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
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

#pragma once

#include "archived_tensor_functions.hpp"
#include "mpo.hpp"
#include <memory>

using namespace std;

namespace block2 {

template <typename S> struct ArchivedMPO : MPO<S> {
    using MPO<S>::n_sites;
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
    void deallocate() override {}
};

} // namespace block2
