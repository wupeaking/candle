use anyhow::{Ok, Result};
use candle::Tensor;
use candle_nn::{, Conv2d, Linear, VarBuilder};

struct Block {}

impl Block {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {})
    }
}

struct ResnetBlock {
    blocks: Vec<Block>,
}

impl ResnetBlock {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self { blocks: vec![] })
    }

    pub fn forward(x: Tensor) -> Result<Tensor> {
        Err(anyhow::anyhow!("unimpl!"))
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
                stride: 3,
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

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        x.max_pool2d(kernel_size, stride)
        let t = Tensor::zeros((1, 100), candle::DType::F32, x.device())?;
        Ok(t)
    }
}
