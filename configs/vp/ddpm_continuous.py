# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Training DDPM with VP SDE."""

from configs.default_configs import get_default_configs



'''
  training TODO:
  training.sde, vp or ms
  training.mask_type 
  training.acs
  training.mean_equal
  training.acc
  sde的std M_hat
  beta_max, beta_min
  num_scales
  -------------------
  sampling TODO:
  training.sde, vp or ms
  training.mask_type 
  training.acs
  training.mean_equal
  training.acc
  sde的std M_hat
  beta_max, beta_min
  ---
  sampling.predictor
  sampling.corrector
  sampling.folder
  sampling.ckpt
  sampling.fft
  sampling.mask_type
  sampling.acc
  sampling.acs
  sde.N
  sde.T
'''
def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'bratsde'
  training.stage = 'mri' 
  training.continuous = True
  training.reduce_mean = True 
  training.joint = True

  # sampling
  sampling = config.sampling
  sampling.num_test = 100
  sampling.batch_size = 1
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama' # reverse_diffusion or euler_maruyama
  sampling.corrector = 'none' # langevin or none

  sampling.folder = '2023_02_22T15_27_19_brats_ddpm_bratsde_mri_N_1000'
  sampling.mri_folder = '2022_10_28T15_43_40_ddpm_mssde_low_frequency_None_16_noequal_N_1000'
  sampling.ckpt = 20 # 
  sampling.acs = '18'
  sampling.snr = 0.075 ##### 
  sampling.mse = 2.5 ##### predictor_mse
  sampling.corrector_mse = 5. ###
  
  # data
  data = config.data
  data.centered = False # True: Input is in [-1, 1]
  data.dataset_name = 'brats'
  data.image_size = 256
  data.normalize_type = 'minmax' # minmax or std
  data.normalize_coeff = 1.5 # normalize coefficient

  # model
  model = config.model
  model.name = 'ddpm'
  model.channel_merge = True 
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  return config
