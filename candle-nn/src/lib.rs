use candle::{Result, Tensor};

pub mod activation;
pub mod conv;
pub mod embedding;
pub mod group_norm;
pub mod init;
pub mod layer_norm;
pub mod linear;
pub mod loss;
pub mod ops;
pub mod optim;
pub mod var_builder;

pub use activation::Activation;
pub use conv::{conv1d, conv2d, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
pub use embedding::{embedding, Embedding};
pub use group_norm::{group_norm, GroupNorm};
pub use init::Init;
pub use layer_norm::{layer_norm, rms_norm, LayerNorm, LayerNormConfig, RmsNorm};
pub use linear::{linear, linear_no_bias, Linear};
pub use optim::{AdamW, ParamsAdamW, SGD};
pub use var_builder::{VarBuilder, VarMap};

// A simple trait defining a module with forward method using a single argument.
pub trait Module: std::fmt::Debug {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;

    /// Change the module to use training mode vs eval mode.
    ///
    /// The default implementation does nothing as this is only used for a couple modules such as
    /// dropout or batch-normalization.
    fn set_training(&mut self, _training: bool) {}
}

impl Module for candle::quantized::QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
