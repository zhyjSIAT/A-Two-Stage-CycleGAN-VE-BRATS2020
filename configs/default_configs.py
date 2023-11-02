import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 1
  training.continue_train = False
  training.continue_folder = '2023_06_02T15_12_28_brats_ncsnpp_bratsde_none_N_1500' # vp 2023_02_20T20_22_47_ddpm_bratsde_mri_N_1000 
  training.continue_ckpt = 10
  training.epochs = 1000
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = False
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16 ##### 
  sampling.mse = 2.5 ##### predictor_mse
  sampling.corrector_mse = 5. ###

  # data
  config.data = data = ml_collections.ConfigDict()
  data.image_size = 256
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 2 
  
  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000 ###
  model.beta_min = 0.1
  model.beta_max = 20. ###
  model.dropout = 0.1 ###
  model.matrix = True ###
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config