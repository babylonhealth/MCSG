
# Copyright 2019 Babylon Partners. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
SHOGUN_DATA_DIR=os.getenv('SHOGUN_DATA_DIR', '../../../data')
import shogun as sg
import numpy as np


# Results on each sub-task when using 128000 bootstrap samples.
# i.e. We sample n-tuples (method_1, method_2, ..., method_n, human_score) 
# 128000 times with replacement for a given task and compute the Spearman
# correlation coeficients for {(method_i, human_score)} using the boostrap samples

f_gauss = np.array([0.4442,0.7994,0.5552,0.6795,0.4925,0.2429,0.7194,0.6386,0.4713,0.6607,0.6463,0.7522,0.7441,0.6659,0.6080,0.7216,0.6771,0.7343,0.8345,0.6577,0.7351,0.8230,0.8405,0.7240])
g_gauss = np.array([0.4455,0.7342,0.5259,0.6628,0.4972,0.3312,0.6994,0.5719,0.4133,0.6635,0.6161,0.7262,0.6986,0.6670,0.5613,0.7206,0.6633,0.7151,0.8101,0.6350,0.7159,0.8170,0.8307,0.6667])
w_gauss = np.array([0.4177,0.7108,0.5980,0.6679,0.5000,0.4016,0.6376,0.6815,0.4375,0.6270,0.6055,0.7348,0.7647,0.6720,0.5525,0.7307,0.7105,0.6752,0.8149,0.5862,0.6769,0.8083,0.8288,0.6793])
f_sif = np.array([0.4311,0.8363,0.5802,0.6477,0.4653,0.4644,0.7059,0.7644,0.4317,0.6929,0.6226,0.7710,0.8095,0.6484,0.6777,0.7244,0.7104,0.7246,0.8305,0.5050,0.7208,0.8252,0.8118,0.7231])
g_sif = np.array([0.4760,0.8015,0.5363,0.5985,0.3470,0.4154,0.6751,0.6473,0.3808,0.6783,0.5772,0.7284,0.7278,0.5361,0.4846,0.6905,0.5952,0.6892,0.7817,0.4974,0.6807,0.7255,0.7489,0.6504])
w_sif = np.array([0.3785,0.8037,0.4629,0.6667,0.4388,0.4364,0.6462,0.7571,0.4678,0.6567,0.5803,0.7582,0.8210,0.6322,0.6287,0.7324,0.7085,0.6769,0.8286,0.4879,0.6788,0.8304,0.7984,0.7026])
f_pca = np.array([0.3778,0.8358,0.5739,0.6522,0.4728,0.5186,0.7133,0.7834,0.4430,0.6737,0.6192,0.7749,0.8280,0.6526,0.6853,0.7156,0.7185,0.7303,0.8411,0.5405,0.7184,0.8348,0.8260,0.7267])
g_pca = np.array([0.3844,0.8131,0.5418,0.6080,0.4565,0.4257,0.6910,0.7623,0.4157,0.6745,0.5895,0.7551,0.8145,0.6136,0.6198,0.6755,0.7115,0.7064,0.8153,0.5578,0.6893,0.7917,0.8195,0.6857])
w_pca = np.array([0.3384,0.8061,0.4549,0.6574,0.4444,0.4664,0.6498,0.7816,0.4667,0.6250,0.5816,0.7630,0.8355,0.6480,0.6482,0.7299,0.7421,0.6814,0.8372,0.5409,0.6789,0.8382,0.8142,0.6996])
f_mwv = np.array([0.4434,0.7395,0.6242,0.6739,0.4594,0.4040,0.7019,0.6619,0.4122,0.6737,0.6204,0.6956,0.7446,0.6627,0.5667,0.7416,0.6518,0.7187,0.7782,0.4946,0.7146,0.7616,0.7848,0.6632])
g_mwv = np.array([0.4717,0.6762,0.5750,0.6135,0.3345,0.3693,0.6371,0.5344,0.3552,0.6500,0.5538,0.6148,0.6444,0.5324,0.3800,0.6823,0.5307,0.6651,0.7198,0.4262,0.6569,0.5659,0.7201,0.5279])
w_mwv = np.array([0.4089,0.7657,0.4973,0.6786,0.4192,0.4144,0.6395,0.7017,0.4435,0.6425,0.5844,0.7450,0.7779,0.6535,0.5144,0.7440,0.6555,0.6728,0.8119,0.4471,0.6782,0.7590,0.7909,0.6535])
f_wmd = np.array([0.5033,0.5333,0.5976,0.6835,0.4227,0.3758,0.6352,0.4001,0.3451,0.6123,0.5805,0.6529,0.5829,0.6891,0.5282,0.7459,0.6103,0.6807,0.7239,0.5964,0.6956,0.7608,0.8016,0.3062])
g_wmd = np.array([0.5001,0.5445,0.5738,0.6695,0.4145,0.3760,0.6373,0.3601,0.3274,0.6176,0.5773,0.6470,0.5530,0.6848,0.4807,0.7388,0.6050,0.6805,0.7196,0.5820,0.6887,0.7433,0.7954,0.2703])
w_wmd = np.array([0.4673,0.6114,0.4725,0.7074,0.4255,0.3943,0.6096,0.4560,0.3884,0.5950,0.5825,0.6817,0.6180,0.6841,0.5058,0.7181,0.6407,0.6631,0.7374,0.5840,0.6736,0.7326,0.8210,0.3545])

f = [f_gauss, f_sif, f_pca, f_mwv, f_wmd]
g = [g_gauss, g_sif, g_pca, g_mwv, g_wmd]
w = [w_gauss, w_sif, w_pca, w_mwv, w_wmd]

rejected = np.zeros((3, 5, 5))
for k, emb in enumerate([f, g, w]):
    for i, X in enumerate(emb):
        for j, Y in enumerate(emb):
            sg.Math.init_random(0)
            # turn data into Shogun representation (columns vectors)
            feat_p=sg.RealFeatures(X.reshape(1,len(X)))
            feat_q=sg.RealFeatures(Y.reshape(1,len(Y)))
            # choose kernel for testing. Here: Gaussian
            kernel_width=1
            kernel=sg.GaussianKernel(10, kernel_width)
            # create mmd instance of test-statistic
            mmd=sg.QuadraticTimeMMD()
            mmd.set_kernel(kernel)
            mmd.set_p(feat_p)
            mmd.set_q(feat_q)
            # compute unbiased test statistic
            mmd.set_statistic_type(sg.ST_UNBIASED_FULL)
            statistic=unbiased_statistic=mmd.compute_statistic()

            mmd.set_null_approximation_method(sg.NAM_PERMUTATION)
            mmd.set_num_null_samples(1000)
            # compute p-value for computed test statistic
            p_value = mmd.compute_p_value(statistic)

            # compute threshold for rejecting H_0 for a given test power
            alpha = 0.05
            threshold = mmd.compute_threshold(alpha)

            if statistic > threshold:
                rejected[k][i][j] = 1

            if p_value < alpha:
                rejected[k][i][j] = 1

            binary_test_result = mmd.perform_test(alpha)
            if binary_test_result:
                rejected[k][i][j] = 1

print(rejected)
