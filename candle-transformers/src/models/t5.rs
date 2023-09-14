// T5 Text Encoder
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Activation, Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

fn default_relative_attention_max_distance() -> usize {
    128
}

fn default_is_decoder() -> bool {
    false
}

fn default_use_cache() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    d_model: usize,
    d_kv: usize,
    d_ff: usize,
    num_layers: usize,
    num_decoder_layers: Option<usize>,
    num_heads: usize,
    relative_attention_num_buckets: usize,
    #[serde(default = "default_relative_attention_max_distance")]
    relative_attention_max_distance: usize,
    dropout_rate: f64,
    layer_norm_epsilon: f64,
    initializer_factor: f64,
    #[serde(default)]
    feed_forward_proj: Activation,
    #[serde(default = "default_is_decoder")]
    is_decoder: bool,
    is_encoder_decoder: bool,
    #[serde(default = "default_use_cache")]
    use_cache: bool,
    pad_token_id: usize,
    eos_token_id: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: Activation::Relu,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
        }
    }
}

impl Config {
    // https://huggingface.co/facebook/musicgen-small/blob/495da4ad086b3416a27c6187f9239f9fd96f3962/config.json#L184
    pub fn musicgen_small() -> Self {
        Self {
            d_ff: 3072,
            d_kv: 64,
            d_model: 768,
            dropout_rate: 0.1,
            eos_token_id: 1,
            feed_forward_proj: Activation::Relu,
            initializer_factor: 1.0,
            is_decoder: false,
            is_encoder_decoder: true,
            layer_norm_epsilon: 1e-6,
            num_decoder_layers: Some(12),
            num_heads: 12,
            num_layers: 12,
            pad_token_id: 0,
            relative_attention_max_distance: 128,
            relative_attention_num_buckets: 32,
            use_cache: true,
            vocab_size: 32128,
        }
    }
}

#[derive(Debug)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
}

impl T5LayerNorm {
    fn load(h: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(h, "weight")?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        // variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct T5DenseActDense {
    wi: Linear,
    wo: Linear,
    act: Activation,
}

impl T5DenseActDense {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("wi"))?;
        let wo = linear_no_bias(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        Ok(Self {
            wi,
            wo,
            act: Activation::Relu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.wi.forward(xs)?;
        let xs = self.act.forward(&xs)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct T5DenseGatedActDense {
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
    act: Activation,
}

impl T5DenseGatedActDense {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi_0 = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("wi_0"))?;
        let wi_1 = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("wi_1"))?;
        let wo = linear_no_bias(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        Ok(Self {
            wi_0,
            wi_1,
            wo,
            act: Activation::NewGelu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden_gelu = self.act.forward(&self.wi_0.forward(xs)?)?;
        let hidden_linear = self.wi_1.forward(xs)?;
        let xs = hidden_gelu.broadcast_mul(&hidden_linear)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct T5LayerFF {
    dense_act: Option<T5DenseActDense>,
    gated_dense_act: Option<T5DenseGatedActDense>,
    layer_norm: T5LayerNorm,
}

impl T5LayerFF {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        let (dense_act, gated_dense_act) = if cfg.feed_forward_proj == Activation::NewGelu {
            (
                None,
                Some(T5DenseGatedActDense::load(vb.pp("DenseReluDense"), cfg)?),
            )
        } else {
            (
                Some(T5DenseActDense::load(vb.pp("DenseReluDense"), cfg)?),
                None,
            )
        };
        Ok(Self {
            dense_act,
            gated_dense_act,
            layer_norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.layer_norm.forward(xs)?;
        let ys = match &self.dense_act {
            Some(dense_act) => dense_act.forward(&ys)?,
            None => self.gated_dense_act.as_ref().unwrap().forward(&ys)?,
        };
        let xs = (xs + ys)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<Embedding>,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    inner_dim: usize,
}

impl T5Attention {
    fn load(h: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        let q = linear_no_bias(cfg.d_model, inner_dim, vb.pp("q"))?;
        let k = linear_no_bias(cfg.d_model, inner_dim, vb.pp("k"))?;
        let v = linear_no_bias(cfg.d_model, inner_dim, vb.pp("v"))?;
        let o = linear_no_bias(inner_dim, cfg.d_model, vb.pp("o"))?;
        let relative_attention_bias = if h {
            let emb = embedding(
                cfg.relative_attention_num_buckets,
                cfg.num_heads,
                vb.pp("relative_attention_bias"),
            )?;
            Some(emb)
        } else {
            None
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            relative_attention_bias,
            relative_attention_num_buckets: cfg.relative_attention_num_buckets,
            relative_attention_max_distance: cfg.relative_attention_max_distance,
            inner_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // TODO: Apply the mask(s)?
        // TODO: kv caching.
        let (b_sz, seq_len) = (xs.dim(0)?, xs.dim(1)?);
        let q = self.q.forward(xs)?;
        let k = self.k.forward(xs)?;
        let v = self.v.forward(xs)?;
        let q = q
            .reshape((b_sz, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let scores = q.matmul(&k.t()?)?;

        let (scores, position_bias) = match position_bias {
            Some(position_bias) => (
                scores.broadcast_add(position_bias)?,
                Some(position_bias.clone()),
            ),
            None => match &self.relative_attention_bias {
                None => (scores, None),
                Some(relative_attention_bias) => {
                    let query_length = seq_len;
                    let key_length = seq_len;
                    // This only handles the bidirectional case.
                    let num_buckets = self.relative_attention_num_buckets as u32 / 2;
                    let max_exact = num_buckets / 2;
                    let relative_position = (0..query_length as u32)
                        .map(|i| {
                            (0..key_length as u32)
                                .map(|j| {
                                    if i < j {
                                        if j - i < max_exact {
                                            j - i + num_buckets
                                        } else {
                                            let b = f32::log(
                                                (j - i) as f32 / max_exact as f32,
                                                self.relative_attention_max_distance as f32
                                                    / max_exact as f32,
                                            ) * (num_buckets - max_exact) as f32;
                                            u32::min(
                                                max_exact + num_buckets + b as u32,
                                                self.relative_attention_num_buckets as u32 - 1,
                                            )
                                        }
                                    } else if i - j < max_exact {
                                        i - j
                                    } else {
                                        let b = f32::log(
                                            (i - j) as f32 / max_exact as f32,
                                            self.relative_attention_max_distance as f32
                                                / max_exact as f32,
                                        ) * (num_buckets - max_exact) as f32;
                                        max_exact + b as u32
                                    }
                                })
                                .collect::<Vec<u32>>()
                        })
                        .collect::<Vec<Vec<_>>>();
                    let relative_buckets = Tensor::new(relative_position, q.device())?;
                    let position_bias = relative_attention_bias
                        .forward(&relative_buckets)?
                        .permute((2, 0, 1))?
                        .unsqueeze(0)?;
                    (scores.broadcast_add(&position_bias)?, Some(position_bias))
                    // TODO: position_bias_masked?
                }
            },
        };

        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.inner_dim))?;
        let attn_output = self.o.forward(&attn_output)?;
        Ok((attn_output, position_bias))
    }
}

#[derive(Debug)]
struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
}

impl T5LayerSelfAttention {
    fn load(h: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attention = T5Attention::load(h, vb.pp("SelfAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        Ok(Self {
            self_attention,
            layer_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed_xs = self.layer_norm.forward(xs)?;
        let (ys, position_bias) = self.self_attention.forward(&normed_xs, position_bias)?;
        let ys = (xs + ys)?;
        Ok((ys, position_bias))
    }
}

#[derive(Debug)]
struct T5LayerCrossAttention {}

impl T5LayerCrossAttention {
    fn load(_vb: VarBuilder, _cfg: &Config) -> Result<Self> {
        todo!()
    }

    fn forward(&self, _xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    cross_attn: Option<T5LayerCrossAttention>,
    ff: T5LayerFF,
}

impl T5Block {
    fn load(has_relative_attention_bias: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let vb = vb.pp("layer");
        let self_attn = T5LayerSelfAttention::load(has_relative_attention_bias, vb.pp("0"), cfg)?;
        let cross_attn = if cfg.is_decoder {
            Some(T5LayerCrossAttention::load(vb.pp("1"), cfg)?)
        } else {
            None
        };
        let ff_i = if cross_attn.is_some() { 2 } else { 1 };
        let ff = T5LayerFF::load(vb.pp(&ff_i.to_string()), cfg)?;
        Ok(Self {
            self_attn,
            cross_attn,
            ff,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (mut xs, position_bias) = self.self_attn.forward(xs, position_bias)?;
        // TODO: clamp for f16?
        if let Some(cross_attn) = &self.cross_attn {
            xs = cross_attn.forward(&xs)?;
            // TODO: clamp for f16?
        }
        let xs = self.ff.forward(&xs)?;
        // TODO: clamp for f16?
        Ok((xs, position_bias))
    }
}

#[derive(Debug)]
struct T5Stack {
    block: Vec<T5Block>,
    shared: Arc<Embedding>,
    final_layer_norm: T5LayerNorm,
}

impl T5Stack {
    fn load(vb: VarBuilder, shared: &Arc<Embedding>, cfg: &Config) -> Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| T5Block::load(i == 0, vb.pp(&format!("block.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let final_layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("final_layer_norm"),
        )?;
        Ok(Self {
            block,
            shared: shared.clone(),
            final_layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let input_embeds = self.shared.as_ref().forward(input_ids)?;
        let mut hidden_states = input_embeds;
        let mut position_bias = None;
        for block in self.block.iter() {
            (hidden_states, position_bias) =
                block.forward(&hidden_states, position_bias.as_ref())?
        }
        self.final_layer_norm.forward(&hidden_states)
    }
}

#[derive(Debug)]
pub struct T5EncoderModel {
    encoder: T5Stack,
    device: Device,
}

impl T5EncoderModel {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let shared = embedding(cfg.vocab_size, cfg.d_model, vb.pp("shared"))?;
        let shared = Arc::new(shared);
        let encoder = T5Stack::load(vb.pp("encoder"), &shared, cfg)?;
        Ok(Self {
            encoder,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.encoder.forward(input_ids)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
