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

# pylint: skip-file

from . import model_utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

from visdom import Visdom
viz = Visdom(port=2012)

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
SelfAttentionBlock = layerspp.SelfAttentionBlock
SpatialAttention = layerspp.SpatialAttention
SPA_PAM_Module = layerspp.SPA_PAM_Module
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
attention_initializer = layerspp

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

@model_utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(model_utils.get_sigmas(config)))

        self.nf = nf = config.model.nf # 128
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks # 2
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult) #[1,1,2,2,2,2,2] len = 7
        self.all_resolutions = all_resolutions = [
            config.data.image_size // (2 ** i) for i in range(num_resolutions)]
        #all_resolutions [256,128,64,32,16,8,4]
        self.conditional = conditional = config.model.conditional  # noise-conditional True
        self.self_attention = self_attention =  config.model.self_attention
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = progressive_input = config.model.progressive_input.lower()
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale # 0.
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            assert config.training.continuous, "Fourier features are only used for continuous training."

            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.model.fourier_scale
            ))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf

        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')
        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
        
        # if self_attention:
            # modules.append(SelfAttentionBlock(in_channels=1, out_channels=64).to(torch.float16))
            # modules.append(SPA_PAM_Module(in_dim=1, out_dim=64).to(torch.float16))
            # modules.append(SpatialAttention(kernel_size=7))
            
        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False)
            ################# TODO #################
            # self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
            # self.pyramid_upsample_last = layerspp.Upsample(out_ch=1, fir=fir, fir_kernel=fir_kernel, with_conv=False)
            ################# TODO #################
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = config.data.num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf)) # channels 2 modules[3]
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level] # ch_mult = (1,1,2,2,2,2,2)
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(
                        in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                        ################# TODO #################
                        # if i_level != 0:
                        #     modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        #     pyramid_ch = channels
                        # else:
                        #     modules.append(conv3x3(in_ch, 1, bias=True, init_scale=init_scale))
                        #     pyramid_ch = 1
                        ################# TODO #################
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(
                            in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        
        #########################################
        if self.config.model.channel_merge:
            modules.append(conv3x3(channels, 1)) # 加了一层卷积
        #########################################

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas)) # 将 std 转换为sin cos形式 (1,128) cat (1,128) ->(1,256)
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb) # (1,512)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None
        # self-attention
        if self.self_attention:
            # self-attention
            sa = SelfAttentionBlock(in_channels=1, out_channels=64).to(torch.float16).cuda()
            # print('attention',x.shape,x[:,0,:,:].shape)
            # viz.images(visualize(torch.squeeze(x[:,0,:,:])), opts=dict(caption='origin_health'))
            x0_ori = visualize(x[:,0,:,:]).to(torch.float16)
            x0 = sa(torch.unsqueeze(x0_ori,0))
            # viz.images(visualize(torch.squeeze(x0)), opts=dict(caption='attention_health'))
            # print('x0',x0.shape)
            
            # viz.images(visualize(torch.squeeze(x[:,1,:,:])), opts=dict(caption='origin_tumor'))
            x1_ori = visualize(x[:,1,:,:]).to(torch.float16)
            # x1 = modules[m_idx](torch.unsqueeze(x1_ori,0)).to(torch.float32)
            attention_map = sa(torch.unsqueeze(x1_ori,0))
            # 再乘一次
            # viz.images(visualize(torch.squeeze(attention_map)), opts=dict(caption='attention_map'))
            x1 = torch.mul(attention_map, x1_ori).to(torch.float32)
            # viz.images(visualize(torch.squeeze(x1)), opts=dict(caption='attention_tumor'))
            
            x[:,0,:,:] = visualize(torch.squeeze(x0))
            x[:,1,:,:] = visualize(torch.squeeze(x1))
            
            # print('after attention',x.shape,x.dtype)
            
            '''
            # spatial attention
            # SAB = SpatialAttention(in_channels=1).cuda()
            # viz.images(visualize(torch.squeeze(x[:,0,:,:])), opts=dict(caption='origin_health'))
            
            # x0_ori = visualize(x[:,0,:,:])
            # x0 = modules[m_idx](torch.unsqueeze(x0_ori,0))
            # # print('atten',x1.shape)
            # viz.images(visualize(torch.squeeze(x0)), opts=dict(caption='attention_health'))
           
            # viz.images(visualize(torch.squeeze(x[:,1,:,:])), opts=dict(caption='origin_tumor'))
            x1_ori = visualize(x[:,1,:,:])
            # x1 = SAB(torch.unsqueeze(x1_ori,0))
            x1 = modules[m_idx](torch.unsqueeze(x1_ori,0))
            # print('atten',x1.shape)
            # viz.images(visualize(torch.squeeze(x1)), opts=dict(caption='attention_tumor'))
            # x[:,0,:,:] = (torch.squeeze(x0))
            x[:,1,:,:] = (torch.squeeze(x1))
            # print('after attention',x.shape,x.dtype)
            m_idx+=1
            '''
        if not self.config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x
        # print('x.shape',x.shape) # torch.Size([1, 2, 256, 256])
        hs = [modules[m_idx](x)]  # weight: [128, 2, 3, 3] modules[3] conv2d(2,128,3,3)
        m_idx += 1
        for i_level in range(self.num_resolutions): # 7
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks): # 2
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions: # 16
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                # debug
                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1] # 取最后一维的数据
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                tmp = hs.pop()
                h = modules[m_idx](torch.cat([h, tmp], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(
                            f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        ###################### TODO ######################
                        # if i_level == 0:
                        #     pyramid = self.pyramid_upsample_last(pyramid)
                        # else:
                        #     pyramid = self.pyramid_upsample(pyramid)
                        ###################### TODO ######################
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(
                            f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h) # h (1,2,256,256)
            m_idx += 1

        ##################################
        if self.config.model.channel_merge:
            h = modules[m_idx](h) # torch.Size([1, 1, 256, 256])
            m_idx += 1
        ##################################

        assert m_idx == len(modules)
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape(
                (x.shape[0], *([1] * len(x.shape[1:])))) # x [1,2,256,256] 
            # debug
            # print(f'used_sigmas: {used_sigmas.shape}') # [1,1,1,1]实际上是一个值
            h = h / used_sigmas
        return h
