mod model;
use anyhow::Result;
use candle::{DType, Tensor, D};
use candle_nn::{ops::softmax, VarBuilder};
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
        // let weights = unsafe { candle::safetensors::MmapedFile::new(weights_file)? };
        // let weights = weights.deserialize()?;
        // let vb = VarBuilder::from_buffered_safetensors(vec![weights], DType::F32, &device);

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], DType::F32, &device)? };
        // from_mmaped_safetensors
        Restnet::load(vb)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = args.build_model()?;
    let device = candle_examples::device(args.cpu)?;

    // 读取图片 转为Tensor
    let img = match args.img {
        Some(img_path) => {
            println!("img_path: {:?}", img_path);
            let img =
                image::open(img_path)?.resize(224, 224, image::imageops::FilterType::Triangle);
            let new_image = img.to_rgb8();
            let img_t = Tensor::new(new_image.as_bytes(), &device)?
                .reshape((
                    new_image.height() as usize,
                    new_image.width() as usize,
                    3_usize,
                ))?
                .permute((2, 0, 1))?; // to [c, h, w]
            (img_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))? // to [b, c, h, w] /255
        }
        None => Tensor::ones((1, 3, 224, 224), DType::F32, &device)?,
    };
    let ret = model.forward(&img)?;
    let pret = softmax(&ret, D::Minus1)?;
    println!("result: {:?}", pret.argmax(D::Minus1)?.to_vec1::<u32>()?);

    // 测试一下性能
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let img = Tensor::randn(0.0f32, 1.0f32, (1, 3, 224, 224), &device)?;
        let ret = model.forward(&img)?;
        println!("{:?}", ret);
    }
    let end = start.elapsed().as_millis();

    println!("累计耗时: {:?}ms 平均耗时: {:?}", end, end / 10000);

    let t = Tensor::ones((1, 3, 2, 2), DType::U8, &candle::Device::Cpu)?;
    let t = t.pad_with_zeros(D::Minus1, 1, 1)?;
    // .pad_with_zeros(D::Minus2, 1, 1)?;
    println!("{:?}", t.get(0)?.get(0)?.to_vec2::<u8>()?);
    Ok(())
}
