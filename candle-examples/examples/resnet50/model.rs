use anyhow::{Ok, Result};
use candle::{Shape, Tensor, D};
use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, Linear, Module, VarBuilder};

struct ConvBN(Conv2d, BatchNorm);
impl ConvBN {
    pub fn new(
        conv_weight: Tensor,
        conv_bias: Option<Tensor>,
        config: Conv2dConfig,
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        bn_weight: Tensor,
        bn_bias: Tensor,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self(
            Conv2d::new(conv_weight, conv_bias, config),
            BatchNorm::new(
                num_features,
                running_mean,
                running_var,
                bn_weight,
                bn_bias,
                eps,
            )?,
        ))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.0.forward(x)?;
        Ok(self.1.forward(&y)?)
    }
}

struct Block {
    conv_bn_01: ConvBN,
    conv_bn_02: ConvBN,
    conv_bn_03: Option<ConvBN>,
    conv_bn_11: ConvBN,
    conv_bn_12: ConvBN,
}

impl Block {
    pub fn load(vb: VarBuilder, i: i32) -> Result<Self> {
        match i {
            1 => {
                let conv_bn_01 = Self::load_conv_bn_helper((64, 64, 3, 3), 1, 1, 64, 0, 1, &vb)?;
                let conv_bn_02 = Self::load_conv_bn_helper((64, 64, 3, 3), 1, 1, 64, 0, 2, &vb)?;
                let conv_bn_11 = Self::load_conv_bn_helper((64, 64, 3, 3), 1, 1, 64, 1, 1, &vb)?;
                let conv_bn_12 = Self::load_conv_bn_helper((64, 64, 3, 3), 1, 1, 64, 1, 2, &vb)?;
                Ok(Self {
                    conv_bn_01,
                    conv_bn_02,
                    conv_bn_03: None,
                    conv_bn_11,
                    conv_bn_12,
                })
            }
            2 => {
                let conv_bn_01 = Self::load_conv_bn_helper((128, 64, 3, 3), 1, 2, 128, 0, 1, &vb)?;
                let conv_bn_02 = Self::load_conv_bn_helper((128, 128, 3, 3), 1, 1, 128, 0, 2, &vb)?;
                let conv_bn_11 = Self::load_conv_bn_helper((128, 128, 3, 3), 1, 1, 128, 1, 1, &vb)?;
                let conv_bn_12 = Self::load_conv_bn_helper((128, 128, 3, 3), 1, 1, 128, 1, 2, &vb)?;
                Ok(Self {
                    conv_bn_01,
                    conv_bn_02,
                    conv_bn_03: Some(ConvBN::new(
                        vb.get((128, 64, 1, 1), "0.downsample.0.weight")?,
                        None,
                        Conv2dConfig {
                            padding: 0,
                            stride: 2,
                            ..Default::default()
                        },
                        128,
                        vb.get(128, "0.downsample.1.running_mean")?,
                        vb.get(128, "0.downsample.1.running_var")?,
                        vb.get(128, "0.downsample.1.weight")?,
                        vb.get(128, "0.downsample.1.bias")?,
                        1e-5,
                    )?),
                    conv_bn_11,
                    conv_bn_12,
                })
            }
            3 => {
                let conv_bn_01 = Self::load_conv_bn_helper((256, 128, 3, 3), 1, 2, 256, 0, 1, &vb)?;
                let conv_bn_02 = Self::load_conv_bn_helper((256, 256, 3, 3), 1, 1, 256, 0, 2, &vb)?;
                let conv_bn_11 = Self::load_conv_bn_helper((256, 256, 3, 3), 1, 1, 256, 1, 1, &vb)?;
                let conv_bn_12 = Self::load_conv_bn_helper((256, 256, 3, 3), 1, 1, 256, 1, 2, &vb)?;
                Ok(Self {
                    conv_bn_01,
                    conv_bn_02,
                    conv_bn_03: Some(ConvBN::new(
                        vb.get((256, 128, 1, 1), "0.downsample.0.weight")?,
                        None,
                        Conv2dConfig {
                            padding: 0,
                            stride: 2,
                            ..Default::default()
                        },
                        256,
                        vb.get(256, "0.downsample.1.running_mean")?,
                        vb.get(256, "0.downsample.1.running_var")?,
                        vb.get(256, "0.downsample.1.weight")?,
                        vb.get(256, "0.downsample.1.bias")?,
                        1e-5,
                    )?),
                    conv_bn_11,
                    conv_bn_12,
                })
            }
            4 => {
                let conv_bn_01 = Self::load_conv_bn_helper((512, 256, 3, 3), 1, 2, 512, 0, 1, &vb)?;
                let conv_bn_02 = Self::load_conv_bn_helper((512, 512, 3, 3), 1, 1, 512, 0, 2, &vb)?;
                let conv_bn_11 = Self::load_conv_bn_helper((512, 512, 3, 3), 1, 1, 512, 1, 1, &vb)?;
                let conv_bn_12 = Self::load_conv_bn_helper((512, 512, 3, 3), 1, 1, 512, 1, 2, &vb)?;
                Ok(Self {
                    conv_bn_01,
                    conv_bn_02,
                    conv_bn_03: Some(ConvBN::new(
                        vb.get((512, 256, 1, 1), "0.downsample.0.weight")?,
                        None,
                        Conv2dConfig {
                            padding: 0,
                            stride: 2,
                            ..Default::default()
                        },
                        512,
                        vb.get(512, "0.downsample.1.running_mean")?,
                        vb.get(512, "0.downsample.1.running_var")?,
                        vb.get(512, "0.downsample.1.weight")?,
                        vb.get(512, "0.downsample.1.bias")?,
                        1e-5,
                    )?),
                    conv_bn_11,
                    conv_bn_12,
                })
            }

            _ => Err(anyhow::anyhow!("unkonw layer")),
        }
        // Ok(Self {})
    }

    fn load_conv_bn_helper<S: Into<Shape>>(
        conv_w_shape: S,
        conv_padding: usize,
        conv_stride: usize,
        num_features: usize,
        layer_index: usize,
        i: usize,
        vb: &VarBuilder,
    ) -> Result<ConvBN> {
        Ok(ConvBN::new(
            vb.get(
                conv_w_shape,
                format!("{}.conv{}.weight", layer_index, i).as_str(),
            )?,
            None,
            Conv2dConfig {
                padding: conv_padding,
                stride: conv_stride,
                ..Default::default()
            },
            num_features,
            vb.get(
                num_features,
                format!("{}.bn{}.running_mean", layer_index, i).as_str(),
            )?,
            vb.get(
                num_features,
                format!("{}.bn{}.running_var", layer_index, i).as_str(),
            )?,
            vb.get(
                num_features,
                format!("{}.bn{}.weight", layer_index, i).as_str(),
            )?,
            vb.get(
                num_features,
                format!("{}.bn{}.bias", layer_index, i).as_str(),
            )?,
            1e-5,
        )?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // let o = self.conv_bn_01
        let y = self.conv_bn_01.forward(x)?.relu()?;
        let y = self.conv_bn_02.forward(&y)?;
        let mut x: Tensor = x.clone();
        if let Some(net) = &self.conv_bn_03 {
            x = net.forward(&x)?;
        }
        let y = y.add(&x)?.relu()?;

        let x = y.clone();
        let y = self.conv_bn_11.forward(&y)?.relu()?;
        let y = self.conv_bn_12.forward(&y)?;
        Ok(x.add(&y)?.relu()?)
    }
}

struct ResnetBlock {
    blocks: Vec<Block>,
}

impl ResnetBlock {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let mut blocks = vec![];
        for i in 1..=4 {
            blocks.push(Block::load(vb.pp(format!("layer{}", i).as_str()), i)?);
        }
        Ok(Self { blocks })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut tmp = x.clone();
        for b in self.blocks.iter() {
            tmp = b.forward(&tmp)?;
        }
        Ok(tmp)
    }
}

pub struct Restnet {
    conv: Conv2d,
    blocks: ResnetBlock,
    linear: Linear,
}

impl Restnet {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        // load conv1 64*3*7*7
        let mut ws = vb.get((64, 3, 7, 7), "conv1.weight")?;
        // 将Conv和BN进行融合  参考这两篇文章
        // https://blog.csdn.net/zengwubbb/article/details/109317661
        // https://cloud.tencent.com/developer/article/1624907
        let bn_mean = vb.get(64, "bn1.running_mean")?;
        let bias = Tensor::zeros(bn_mean.shape(), ws.dtype(), ws.device())?;
        let bn_w = vb.get(64, "bn1.weight")?;
        let bn_bias = vb.get(64, "bn1.bias")?;
        let running_var = vb.get(64, "bn1.running_var")?;
        let var_sqrt = (running_var + 10e-5)?.sqrt()?;
        // 需要调整Tensor的shape
        let a = bn_w.div(&var_sqrt)?.reshape((64, 1, 1, 1))?;
        ws = ws.broadcast_mul(&a)?;
        let bias = bias
            .sub(&bn_mean)?
            .div(&var_sqrt)?
            .mul(&bn_w)?
            .add(&bn_bias)?;

        let conv2d = Conv2d::new(
            ws,
            Some(bias),
            candle_nn::Conv2dConfig {
                padding: 3,
                stride: 2,
                ..Default::default()
            },
        );

        let ws = vb.get((1000, 512), "fc.weight")?;
        let bias = vb.get(1000, "fc.bias")?;
        let linear = Linear::new(ws, Some(bias));

        Ok(Self {
            conv: conv2d,
            blocks: ResnetBlock::load(vb)?,
            linear,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // println!("x: {:?}", x);
        // let padiing_x = x
        //     .pad_with_zeros(D::Minus1, 3, 3)?
        //     .pad_with_zeros(D::Minus2, 3, 3)?;

        let y = self.conv.forward(x)?.relu()?;
        // println!("y: {:?}", y);
        // 填充padding
        let y = y
            .pad_with_zeros(D::Minus1, 1, 1)?
            .pad_with_zeros(D::Minus2, 1, 1)?;
        let y = y.max_pool2d((3, 3), (2, 2))?;
        // println!("y: {:?}", y);
        let y = self.blocks.forward(&y)?;
        // println!("y: {:?}", y);
        let y = y.avg_pool2d((7, 7), (1, 1))?;
        let y = y.flatten_from(1)?;
        Ok(self.linear.forward(&y)?)
    }
}
