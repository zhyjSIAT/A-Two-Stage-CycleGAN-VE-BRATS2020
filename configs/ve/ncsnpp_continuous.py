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
"""Training NCSN++ on Church with VE SDE."""

from configs.default_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'bratsde'
  training.stage = 'none' 
  training.joint = True ###
  training.continuous = True
  training.reduce_mean = True
  
  # sampling
  sampling = config.sampling
  sampling.num_test = 2
  sampling.batch_size = 1
  sampling.method = 'pc' # pc / ode
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin' # langevin
  sampling.folder = '2023_02_18T16_14_58_ncsnpp_bratsde_none_N_1000' 
  
  sampling.mri_folder = '2022_11_28T12_12_19_ncsnpp_petsde_none_N_1000'
  sampling.ckpt = 30
  sampling.mask_type = 'uniform' # uniform, random_uniform or center
  sampling.acc = '8'
  sampling.acs = '18'
  sampling.fft = 'nofft' # fft or nofft
  sampling.snr = 0.075 ##### 
  sampling.mse = 2.5 ##### predictor_mse
  sampling.corrector_mse = 5. ###
  
  # data
  data = config.data
  data.centered = False
  data.dataset_name = 'brats'  #  brats flair t2 inhouse
  
  data.image_size = 256
  data.normalize_type = 'minmax' 
  data.normalize_coeff = 1.5 # normalize coefficient

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.channel_merge = True 
  model.dropout = 0.
  model.sigma_max = 348
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.self_attention = False
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config