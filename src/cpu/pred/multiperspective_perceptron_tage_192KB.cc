/*
 * Copyright 2019 Texas A&M University
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Daniel A. Jiménez
 *  Adapted to gem5 by: Javier Bueno Hedo
 *
 */

/*
 * Multiperspective Perceptron Predictor with TAGE (by Daniel A. Jiménez)
 * 64 KB version
 */

#include "cpu/pred/multiperspective_perceptron_tage_192KB.hh"

namespace gem5
{

namespace branch_prediction
{

MultiperspectivePerceptronTAGE192KB::MultiperspectivePerceptronTAGE192KB(
    const MultiperspectivePerceptronTAGE192KBParams &p)
    : MultiperspectivePerceptronTAGE(p)
{}

void
MultiperspectivePerceptronTAGE192KB::createSpecs()
{
    addSpec(new BLURRYPATH(5, 15, -1, 2.25, 0, 6, *this));
    addSpec(new BLURRYPATH(8, 10, -1, 2.25, 0, 6, *this));
    addSpec(new RECENCYPOS(31, 3.5, 0, 6, *this));
    addSpec(new GHISTMODPATH(3, 7, 1, 2.24, 0, 6, *this));
    addSpec(new MODPATH(3, 20, 3, 2.24, 0, 6, *this));
    addSpec(new IMLI(1, 2.23, 0, 6, *this));
    addSpec(new IMLI(4, 1.98, 0, 6, *this));
    addSpec(new RECENCY(9, 3, -1, 2.51, 0, 6, *this));
    addSpec(new ACYCLIC(12, -1, -1, 2.0, 0, 6, *this));
}

} // namespace branch_prediction
} // namespace gem5
