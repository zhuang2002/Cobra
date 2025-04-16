# Copyright 2024 The HuggingFace Team. All rights reserved.
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
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('torch_npu not found')
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from typing import List
from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import is_torch_version, logging
from ..attention import BasicTransformerBlock, CausalTransformerKVcacheBlock
from ..attention_processor import Attention, AttentionProcessor, AttnProcessor, FusedAttnProcessor2_0
from ..embeddings import PatchEmbed, PixArtAlphaTextProjection, CausalPatchEmbed
from ..modeling_outputs import Transformer2DModelOutput, CausalTransformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormSingle
from ...loaders import PeftAdapterMixin
from einops import rearrange


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_positional_encoding_1d(d_model, max_len=5000):
    """
    生成1D的正弦和余弦位置编码矩阵。

    参数:
        d_model (int): 模型的维度。
        max_len (int): 序列的最大长度。

    返回:
        pe (torch.Tensor): 位置编码矩阵，形状为 (max_len, d_model)。
    """
    # 创建位置编码矩阵
    pe = torch.zeros(max_len, d_model)
    
    # 生成位置索引
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    # 生成频率项
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # 计算正弦和余弦编码
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦
    
    return pe

class CausalSparseDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    A 2D Transformer model as introduced in PixArt family of models (https://arxiv.org/abs/2310.00426,
    https://arxiv.org/abs/2403.04692).

    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        cross_attention_dim (int, optional):
            The dimensionality for cross-attention layers, typically matching the encoder's hidden dimension.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 128):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        interpolation_scale (int, optional): Scale factor to use during interpolating the position embeddings.
        use_additional_conditions (bool, optional): If we're using additional conditions as inputs.
        attention_type (str, optional, defaults to "default"): Kind of attention mechanism to be used.
        caption_channels (int, optional, defaults to None):
            Number of channels to use for projecting the caption embeddings.
        use_linear_projection (bool, optional, defaults to False):
            Deprecated argument. Will be removed in a future version.
        num_vector_embeds (bool, optional, defaults to False):
            Deprecated argument. Will be removed in a future version.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["CausalTransformerKVcacheBlock", "CausalPatchEmbed"]
# {
#   "_class_name": "Transformer2DModel",
#   "_diffusers_version": "0.22.0.dev0",
#   "activation_fn": "gelu-approximate",
#   "attention_bias": true,
#   "attention_head_dim": 72,
#   "attention_type": "default",
#   "caption_channels": 4096,
#   "cross_attention_dim": 1152,
#   "double_self_attention": false,
#   "dropout": 0.0,
#   "in_channels": 4,
#   "norm_elementwise_affine": false,
#   "norm_eps": 1e-06,
#   "norm_num_groups": 32,
#   "norm_type": "ada_norm_single",
#   "num_attention_heads": 16,
#   "num_embeds_ada_norm": 1000,
#   "num_layers": 28,
#   "num_vector_embeds": null,
#   "only_cross_attention": false,
#   "out_channels": 8,
#   "patch_size": 2,
#   "sample_size": 128,
#   "upcast_attention": false,
#   "use_linear_projection": false
# }
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = 8,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        sample_size: int = 128,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        use_additional_conditions: Optional[bool] = None,
        caption_channels: Optional[int] = None,
        attention_type: Optional[str] = "default",
    ):
        super().__init__()

        # Validate inputs.
        if norm_type != "ada_norm_single":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_single" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        if use_additional_conditions is None:
            if sample_size == 128:
                use_additional_conditions = True
            else:
                use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions
        # print(f"----------------self.use_additional_conditions: {self.use_additional_conditions} {sample_size}")

        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = self.config.sample_size
        self.width = self.config.sample_size

        interpolation_scale = (
            self.config.interpolation_scale
            if self.config.interpolation_scale is not None
            else max(self.config.sample_size // 64, 1)
        )
        self.pos_embed = CausalPatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CausalTransformerKVcacheBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels)

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=self.use_additional_conditions
        )
        self.caption_projection = None
        if self.config.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.config.caption_channels, hidden_size=self.inner_dim
            )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.

        Safe to just use `AttnProcessor()` as PixArt doesn't have any exotic attention processors in default model.
        """
        self.set_attn_processor(AttnProcessor())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor, # (b, c, h, w)
        ref_hidden_states: torch.Tensor = None, # (b, n_ref, c, h, w)
        n_ref_lists: List[List[int]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        control_list: Optional[list] = None,
        return_dict: bool = True,
        repa_layer_idx: Optional[int] = None,
        K_cache: Optional[list] = None,
        V_cache: Optional[list] = None,
    ):
        """
        The [`PixArtTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep (`torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            added_cond_kwargs: (`Dict[str, Any]`, *optional*): Additional conditions to be used as inputs.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        no_cache = K_cache is None and V_cache is None

        # 1. Input
        batch_size = hidden_states.shape[0]
        if no_cache:
            b_list = len(n_ref_lists)
            if b_list!=batch_size:
                raise ValueError(f"Number of reference images ({b_list}) does not match the batch size ({batch_size})")
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        
        if no_cache:
            K_cache_cur = []
            V_cache_cur = []
            N_ref = ref_hidden_states.shape[1]
            if N_ref != sum(n_ref_lists[0]):
                raise ValueError(f"Number of reference images ({N_ref}) does not match the sum of reference image counts ({sum(n_ref_lists[0])})")

            height_refs, width_refs = (
                ref_hidden_states.shape[-2] // self.config.patch_size,
                ref_hidden_states.shape[-1] // self.config.patch_size,
            )
            # print('height_refs',height_refs, 'width_refs',width_refs, 'height',height, 'width',width)

            if height_refs*2 != height or width_refs*2 != width:
                raise ValueError(f"Height and width of ref_hidden_states must be twice the size of hidden_states. Got height_refs={height_refs}, width_refs={width_refs}, height={height}, width={width}")

            ref_hidden_states = rearrange(ref_hidden_states, 'b n_ref c h w -> (b n_ref) c h w')
            ref_hidden_states = self.pos_embed(ref_hidden_states)
            ref_hidden_states = rearrange(ref_hidden_states, '(b n_ref) l c -> b n_ref l c', n_ref=N_ref)
        

        hidden_states = self.pos_embed(hidden_states) # (b, l, c)
        # print('hidden_states',hidden_states.shape)

        pos_all = self.pos_embed.get_pos_embed(height, width).to(hidden_states.device)
        # print('pos_all',pos_all.shape)


        pos_all = rearrange(pos_all, 'b (h w) c -> b h w c', h=height, w=width*2)
        pos_all = rearrange(pos_all, 'b (ph h) (pw w) c -> b (ph pw) h w c', ph=2, pw=4)

        if no_cache:
            pos_idx0 = rearrange(pos_all[:,0,:,:,:], 'b h w c -> b (h w) c')
            pos_idx1 = rearrange(pos_all[:,3,:,:,:], 'b h w c -> b (h w) c')
            pos_idx2 = rearrange(pos_all[:,4,:,:,:], 'b h w c -> b (h w) c')
            pos_idx3 = rearrange(pos_all[:,7,:,:,:], 'b h w c -> b (h w) c')
        
        pos_center = rearrange(rearrange(pos_all[:,[1,2,5,6],:,:,:], 'b (ph pw) h w c -> b (ph h) (pw w) c', ph=2, pw=2), 'b h w c -> b (h w) c')

        hidden_states = (hidden_states + pos_center).to(dtype=hidden_states.dtype)


        if no_cache:

            pos_repeat_list = []
            for ref_b_idx in range(b_list):
                pos_idx0_repeat = None
                pos_idx1_repeat = None
                pos_idx2_repeat = None
                pos_idx3_repeat = None
                cur_n_ref_list = n_ref_lists[ref_b_idx]
                if cur_n_ref_list[0] > 0:
                    pos_idx0_repeat = pos_idx0.unsqueeze(0).repeat(1, cur_n_ref_list[0], 1, 1)
                if cur_n_ref_list[1] > 0:
                    pos_idx1_repeat = pos_idx1.unsqueeze(0).repeat(1, cur_n_ref_list[1], 1, 1)
                if cur_n_ref_list[2] > 0:
                    pos_idx2_repeat = pos_idx2.unsqueeze(0).repeat(1, cur_n_ref_list[2], 1, 1)
                if cur_n_ref_list[3] > 0:
                    pos_idx3_repeat = pos_idx3.unsqueeze(0).repeat(1, cur_n_ref_list[3], 1, 1)

                pos_repeat = []
                if cur_n_ref_list[0] > 0:
                    pos_repeat.append(pos_idx0_repeat)
                if cur_n_ref_list[1] > 0:
                    pos_repeat.append(pos_idx1_repeat)
                if cur_n_ref_list[2] > 0:
                    pos_repeat.append(pos_idx2_repeat)
                if cur_n_ref_list[3] > 0:
                    pos_repeat.append(pos_idx3_repeat)

                pos_repeat = torch.cat(pos_repeat, dim=1)
                pos_repeat_list.append(pos_repeat)

        
            pos_repeat_all = torch.cat(pos_repeat_list, dim=0)
        
            ref_hidden_states = (ref_hidden_states + pos_repeat_all).to(dtype=hidden_states.dtype)

            
            timestep_ref, _ = self.adaln_single(
                timestep*0, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            # print('timestep_ref',timestep*0, timestep)

        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # 2. Blocks
        # add
        control_idx = 0
        cache_idx = 0

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                if no_cache:
                    hidden_states, ref_hidden_states, K_sample, V_sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        ref_hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        cross_attention_kwargs,
                        None,
                        K_cache = None,
                        V_cache = None,
                        timestep_ref = timestep_ref,
                        **ckpt_kwargs,
                    )
                    K_cache_cur.append(K_sample)
                    V_cache_cur.append(V_sample)
                else:
                    hidden_states, _, _, _ = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        None,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        cross_attention_kwargs,
                        None,
                        K_cache = K_cache[cache_idx],
                        V_cache = V_cache[cache_idx],
                        timestep_ref = None,
                        **ckpt_kwargs,
                    )
            else:
                if no_cache:
                    hidden_states, ref_hidden_states, K_sample, V_sample = block(
                        hidden_states,
                        ref_hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=None,
                        K_cache = None,
                        V_cache = None,
                        timestep_ref = timestep_ref,
                    )
                    K_cache_cur.append(K_sample)
                    V_cache_cur.append(V_sample)
                else:
                    hidden_states, _, _, _ = block(
                        hidden_states,
                        None,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        K_cache = K_cache[cache_idx],
                        V_cache = V_cache[cache_idx],
                        timestep_ref = None,
                    )

                
            if control_list!=None and control_idx<len(control_list):
                hidden_states += control_list[control_idx][0]
                
                control_idx+=1

            cache_idx+=1
                
        # 3. Output
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)

        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        # print('hidden_states.shape', hidden_states.shape)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
        )

        if no_cache:
            return (output, K_cache_cur, V_cache_cur)
        else:
            return (output, None, None)


