mod model;
use anyhow::{anyhow, Error as E, Result};
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use image::{self, EncodableLayout};
use model::Restnet;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// img path
    #[arg(long)]
    img: Option<String>,

    #[arg(long, default_value = "resnet18.safetensors")]
    weight_file: Option<String>,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,
}

impl Args {
    fn build_model(&self) -> Result<Restnet> {
        let device = candle_examples::device(self.cpu)?;
        let weights_file = self.weight_file.as_ref().expect("need weighs file");
        let weights = unsafe { candle::safetensors::MmapedFile::new(weights_file)? };
        let weights = weights.deserialize()?;
        let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, &device);
        Restnet::load(vb)
    }
}

fn main() -> Result<()> {
    let start = std::time::Instant::now();
    let args = Args::parse();
    let model = args.build_model()?;
    let device = candle_examples::device(args.cpu)?;

    // 读取图片 转为Tensor
    let img = match args.img {
        Some(img_path) => {
            let img =
                image::open(img_path)?.resize(224, 224, image::imageops::FilterType::Triangle);
            let new_image = img.to_rgb8();
            Tensor::new(new_image.as_bytes(), &device)?
                .reshape((
                    3_usize,
                    new_image.height() as usize,
                    new_image.width() as usize,
                ))?
                .to_dtype(DType::F32)?
        }
        None => Tensor::ones((1, 3, 224, 224), DType::F32, &device)?,
    };

    let ret = model.forward(img)?;
    println!("{:?}", ret);
    Ok(())
}
