pub mod coco_classes;
pub mod imagenet;

use candle::{Device, Result, Tensor};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(device)
    }
}

pub fn load_image<P: AsRef<std::path::Path>>(
    p: P,
    resize_longest: Option<usize>,
) -> Result<(Tensor, usize, usize)> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?;
    let (initial_h, initial_w) = (img.height() as usize, img.width() as usize);
    let img = match resize_longest {
        None => img,
        Some(resize_longest) => {
            let (height, width) = (img.height(), img.width());
            let resize_longest = resize_longest as u32;
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            img.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        }
    };
    let (height, width) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    Ok((data, initial_h, initial_w))
}

pub fn load_image_and_resize<P: AsRef<std::path::Path>>(
    p: P,
    width: usize,
    height: usize,
) -> Result<Tensor> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let data = img.into_raw();
    Tensor::from_vec(data, (width, height, 3), &Device::Cpu)?.permute((2, 0, 1))
}

/// Saves an image to disk using the image crate, this expects an input with shape
/// (c, height, width).
pub fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        candle::bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => candle::bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(candle::Error::wrap)?;
    Ok(())
}

pub fn save_image_resize<P: AsRef<std::path::Path>>(
    img: &Tensor,
    p: P,
    h: usize,
    w: usize,
) -> Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        candle::bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => candle::bail!("error saving image {p:?}"),
        };
    let image = image::DynamicImage::from(image);
    let image = image.resize_to_fill(w as u32, h as u32, image::imageops::FilterType::CatmullRom);
    image.save(p).map_err(candle::Error::wrap)?;
    Ok(())
}
