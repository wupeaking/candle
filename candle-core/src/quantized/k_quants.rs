use super::utils::{
    get_scale_min_k4, group_for_dequantization, group_for_quantization, make_q3_quants,
    make_qkx1_quants, make_qx_quants, nearest_int,
};
use super::GgmlDType;
use crate::Result;
use half::f16;
use rayon::prelude::*;

// Default to QK_K 256 rather than 64.
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

pub trait GgmlType: Sized + Clone + Send + Sync {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;
    type VecDotType: GgmlType;

    // This is only safe for types that include immediate values such as float/int/...
    fn zeros() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()>;
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()>;

    /// Dot product used as a building block for quantized mat-mul.
    /// n is the number of elements to be considered.
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32>;
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [u8; QK4_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) d: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub(crate) d: f16,
    pub(crate) s: f16,
    pub(crate) qs: [u8; QK8_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2K {
    pub(crate) scales: [u8; QK_K / 16],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f16,
    pub(crate) dmin: f16,
}
const _: () = assert!(QK_K / 16 + QK_K / 4 + 2 * 2 == std::mem::size_of::<BlockQ2K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ3K {
    pub(crate) hmask: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) scales: [u8; 12],
    pub(crate) d: f16,
}
const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3K>());

#[derive(Debug, Clone, PartialEq)]
// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
#[repr(C)]
pub struct BlockQ4K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qh: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () =
    assert!(QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE == std::mem::size_of::<BlockQ5K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ6K {
    pub(crate) ql: [u8; QK_K / 2],
    pub(crate) qh: [u8; QK_K / 4],
    pub(crate) scales: [i8; QK_K / 16],
    pub(crate) d: f16,
}
const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}
const _: () = assert!(4 + QK_K + QK_K / 16 * 2 == std::mem::size_of::<BlockQ8K>());

impl GgmlType for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLCK_SIZE: usize = QK4_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1525
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        let qk = Self::BLCK_SIZE;
        if k % qk != 0 {
            crate::bail!("dequantize_row_q4_0: {k} is not divisible by {qk}")
        }

        let nb = k / qk;
        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..(qk / 2) {
                let x0 = (xs[i].qs[j] & 0x0F) as i16 - 8;
                let x1 = (xs[i].qs[j] >> 4) as i16 - 8;

                ys[i * qk + j] = (x0 as f32) * d;
                ys[i * qk + j + qk / 2] = (x1 as f32) * d;
            }
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q4_0
        let qk = Self::BLCK_SIZE;
        let k = xs.len();
        if k % qk != 0 {
            crate::bail!("{k} is not divisible by {}", qk);
        };
        let nb = k / qk;
        if ys.len() != nb {
            crate::bail!("size mismatch {} {} {}", xs.len(), ys.len(), qk,)
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;

            let xs = &xs[i * qk..(i + 1) * qk];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -8.0;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);

            for (j, q) in ys.qs.iter_mut().enumerate() {
                let x0 = xs[j] * id;
                let x1 = xs[qk / 2 + j] * id;
                let xi0 = u8::min(15, (x0 + 8.5) as u8);
                let xi1 = u8::min(15, (x1 + 8.5) as u8);
                *q = xi0 | (xi1 << 4)
            }
        }
        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L2361C10-L2361C122
    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q4_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q4_0_q8_0(n, xs, ys);

        let qk = QK8_0;
        let nb = n / qk;
        if n % QK8_0 != 0 {
            crate::bail!("vec_dot_q4_0_q8_0: {n} is not divisible by {qk}")
        }
        if nb % 2 != 0 {
            crate::bail!("vec_dot_q4_0_q8_0: {nb} is not even")
        }

        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sum_i = 0;
            for j in 0..qk / 2 {
                let v0 = (xs.qs[j] & 0x0F) as i32 - 8;
                let v1 = (xs.qs[j] >> 4) as i32 - 8;
                sum_i += v0 * ys.qs[j] as i32 + v1 * ys.qs[j + qk / 2] as i32
            }
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        Ok(sumf)
    }
}

impl GgmlType for BlockQ4_1 {
    const DTYPE: GgmlDType = GgmlDType::Q4_1;
    const BLCK_SIZE: usize = QK4_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1545
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK4_1 != 0 {
            crate::bail!("dequantize_row_q4_1: {k} is not divisible by {QK4_1}");
        }

        let nb = k / QK4_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();

            for j in 0..(QK4_1 / 2) {
                let x0 = xs[i].qs[j] & 0x0F;
                let x1 = xs[i].qs[j] >> 4;

                ys[i * QK4_1 + j] = (x0 as f32) * d + m;
                ys[i * QK4_1 + j + QK4_1 / 2] = (x1 as f32) * d + m;
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ5_0 {
    const DTYPE: GgmlDType = GgmlDType::Q5_0;
    const BLCK_SIZE: usize = QK5_0;
    type VecDotType = BlockQ8_0;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1566
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK5_0 != 0 {
            crate::bail!("dequantize_row_q5_0: {k} is not divisible by {QK5_0}");
        }

        let nb = k / QK5_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let qh: u32 = unsafe { std::mem::transmute_copy(&xs[i].qh) };

            for j in 0..(QK5_0 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = ((xs[i].qs[j] & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((xs[i].qs[j] >> 4) | xh_1) as i32 - 16;

                ys[i * QK5_0 + j] = (x0 as f32) * d;
                ys[i * QK5_0 + j + QK5_0 / 2] = (x1 as f32) * d;
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ5_1 {
    const DTYPE: GgmlDType = GgmlDType::Q5_1;
    const BLCK_SIZE: usize = QK5_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1592
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK5_1 != 0 {
            crate::bail!("dequantize_row_q5_1: {k} is not divisible by {QK5_1}");
        }

        let nb = k / QK5_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();
            let qh: u32 = unsafe { std::mem::transmute_copy(&xs[i].qh) };

            for j in 0..(QK5_1 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = (xs[i].qs[j] & 0x0F) | xh_0;
                let x1 = (xs[i].qs[j] >> 4) | xh_1;

                ys[i * QK5_1 + j] = (x0 as f32) * d + m;
                ys[i * QK5_1 + j + QK5_1 / 2] = (x1 as f32) * d + m;
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ8_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK8_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1619
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK8_0 != 0 {
            crate::bail!("dequantize_row_q8_0: {k} is not divisible by {QK8_0}");
        }

        let nb = k / QK8_0;

        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..QK8_0 {
                ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
            }
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q8_0
        let k = xs.len();
        if k % Self::BLCK_SIZE != 0 {
            crate::bail!("{k} is not divisible by {}", Self::BLCK_SIZE);
        };
        let nb = k / Self::BLCK_SIZE;
        if ys.len() != nb {
            crate::bail!(
                "size mismatch {} {} {}",
                xs.len(),
                ys.len(),
                Self::BLCK_SIZE
            )
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
                *y = f32::round(x * id) as i8
            }
        }
        Ok(())
    }

    fn vec_dot(_: usize, _: &[Self], _: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }
}

impl GgmlType for BlockQ8_1 {
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_1;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
    fn to_float(_xs: &[Self], _ys: &mut [f32]) -> Result<()> {
        todo!()
    }
}

impl GgmlType for BlockQ2K {
    const DTYPE: GgmlDType = GgmlDType::Q2K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L279
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        const Q4SCALE: f32 = 15.0;

        for (block, x) in group_for_quantization(xs, ys)? {
            //calculate scales and mins
            let mut mins: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];

            for (j, x_scale_slice) in x.chunks(16).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(3, 5, x_scale_slice);
            }
            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            if max_scale > 0.0 {
                let iscale = Q4SCALE / max_scale;
                for (j, scale) in scales.iter().enumerate().take(QK_K / 16) {
                    block.scales[j] = nearest_int(iscale * scale) as u8;
                }
                block.d = f16::from_f32(max_scale / Q4SCALE);
            } else {
                for j in 0..QK_K / 16 {
                    block.scales[j] = 0;
                }
                block.d = f16::from_f32(0.0);
            }

            if max_min > 0.0 {
                let iscale = Q4SCALE / max_min;
                for (j, scale) in block.scales.iter_mut().enumerate() {
                    let l = nearest_int(iscale * mins[j]) as u8;
                    *scale |= l << 4;
                }
                block.dmin = f16::from_f32(max_min / Q4SCALE);
            } else {
                block.dmin = f16::from_f32(0.0);
            }

            let mut big_l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let d = block.d.to_f32() * (block.scales[j] & 0xF) as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * (block.scales[j] >> 4) as f32;
                for ii in 0..16 {
                    let ll = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3);
                    big_l[16 * j + ii] = ll as u8;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for ll in 0..32 {
                    block.qs[j / 4 + ll] = big_l[j + ll]
                        | (big_l[j + ll + 32] << 2)
                        | (big_l[j + ll + 64] << 4)
                        | (big_l[j + ll + 96] << 6);
                }
            }
        }
        Ok(())
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L354
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        for (block, y) in group_for_dequantization(xs, ys)? {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();

            let mut is = 0;

            for (y_block, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
                // Step by 32 over q.
                let mut shift = 0;
                let mut y_block_index = 0;
                for _j in 0..4 {
                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[..16] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[16..] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    shift += 2;
                }
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ3K {
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        for (block, x) in group_for_quantization(xs, ys)? {
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];
            for (j, x_scale_slice) in x.chunks_exact(16).enumerate() {
                scales[j] = make_q3_quants(x_scale_slice, 4, true);
            }

            // Get max scale by absolute value.
            let max_scale = scales
                .iter()
                .fold(0.0, |max, &val| if val.abs() > max { val } else { max });

            block.scales.fill(0);

            if max_scale != 0.0 {
                let iscale = -32.0 / max_scale;
                for (j, scale) in scales.iter().enumerate() {
                    let l_val = nearest_int(iscale * scale);
                    let l_val = l_val.clamp(-32, 31) + 32;
                    if j < 8 {
                        block.scales[j] = (l_val & 0xF) as u8;
                    } else {
                        block.scales[j - 8] |= ((l_val & 0xF) << 4) as u8;
                    }
                    let l_val = l_val >> 4;
                    block.scales[j % 4 + 8] |= (l_val << (2 * (j / 4))) as u8;
                }
                block.d = f16::from_f32(1.0 / iscale);
            } else {
                block.d = f16::from_f32(0.0);
            }

            let mut l: [i8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let sc = if j < 8 {
                    block.scales[j] & 0xF
                } else {
                    block.scales[j - 8] >> 4
                };
                let sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    for ii in 0..16 {
                        let l_val = nearest_int(x[16 * j + ii] / d);
                        l[16 * j + ii] = (l_val.clamp(-4, 3) + 4) as i8;
                    }
                }
            }

            block.hmask.fill(0);
            let mut m = 0;
            let mut hm = 1;

            for ll in l.iter_mut() {
                if *ll > 3 {
                    block.hmask[m] |= hm;
                    *ll -= 4;
                }
                m += 1;
                if m == QK_K / 8 {
                    m = 0;
                    hm <<= 1;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for l_val in 0..32 {
                    block.qs[j / 4 + l_val] = (l[j + l_val]
                        | (l[j + l_val + 32] << 2)
                        | (l[j + l_val + 64] << 4)
                        | (l[j + l_val + 96] << 6))
                        as u8;
                }
            }
        }

        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        for (block, y) in group_for_dequantization(xs, ys)? {
            //Reconstruct the scales
            let mut aux = [0; 4];
            let aux_raw = unsafe {
                std::mem::transmute::<&mut [u8; 12], &mut [u32; 3]>(&mut block.scales.clone())
            };
            aux[0..3].copy_from_slice(aux_raw);

            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

            //Transfer the scales into an i8 array
            let scales: &mut [i8] =
                unsafe { std::slice::from_raw_parts_mut(aux.as_mut_ptr() as *mut i8, 16) };

            let d_all = block.d.to_f32();
            let mut m = 1;
            let mut is = 0;
            let mut dl;

            // Dequantize both 128 long blocks
            // 32 qs values per 128 long block
            // Each 16 elements get a scale
            for (y, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
                let mut shift = 0;
                for shift_scoped_y in y.chunks_exact_mut(32) {
                    for (scale_index, scale_scoped_y) in
                        shift_scoped_y.chunks_exact_mut(16).enumerate()
                    {
                        dl = d_all * (scales[is] as f32 - 32.0);
                        for (i, inner_y) in scale_scoped_y.iter_mut().enumerate() {
                            let new_y = dl
                                * (((qs[i + 16 * scale_index] >> shift) & 3) as i8
                                    - if (block.hmask[i + 16 * scale_index] & m) == 0 {
                                        4
                                    } else {
                                        0
                                    }) as f32;
                            *inner_y = new_y;
                        }
                        // 16 block finished => advance scale index
                        is += 1;
                    }
                    // 32 block finished => increase shift and m
                    shift += 2;
                    m <<= 1;
                }
            }
        }

        Ok(())
    }
}

impl GgmlType for BlockQ4K {
    const DTYPE: GgmlDType = GgmlDType::Q4K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        for (block, x) in group_for_quantization(xs, ys)? {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(15, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

            for j in 0..QK_K / 32 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }

            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                    }
                }
            }

            let q = &mut block.qs;
            for j in (0..QK_K).step_by(64) {
                for l_val in 0..32 {
                    let offset_index = (j / 64) * 32 + l_val;
                    q[offset_index] = l[j + l_val] | (l[j + l_val + 32] << 4);
                }
            }
        }
        Ok(())
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L735
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        for (block, y) in group_for_dequantization(xs, ys)? {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let q = &block.qs;
            let mut is = 0;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let q = &q[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for q in q {
                    y[ys_index] = d1 * (q & 0xF) as f32 - m1;
                    ys_index += 1;
                }
                for q in q {
                    y[ys_index] = d2 * (q >> 4) as f32 - m2;
                    ys_index += 1;
                }
                is += 2;
            }
        }
        Ok(())
    }
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
impl GgmlType for BlockQ5K {
    const DTYPE: GgmlDType = GgmlDType::Q5K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L793
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        for (block, x) in group_for_quantization(xs, ys)? {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(31, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
            for j in 0..QK_K / 32 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }
            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];
            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * m as f32;
                for ii in 0..32 {
                    let ll = nearest_int((x[32 * j + ii] + dm) / d);
                    l[32 * j + ii] = ll.clamp(0, 31) as u8;
                }
            }

            let qh = &mut block.qh;
            let ql = &mut block.qs;
            qh.fill(0);

            let mut m1 = 1;
            let mut m2 = 2;
            for n in (0..QK_K).step_by(64) {
                let offset = (n / 64) * 32;
                for j in 0..32 {
                    let mut l1 = l[n + j];
                    if l1 > 15 {
                        l1 -= 16;
                        qh[j] |= m1;
                    }
                    let mut l2 = l[n + j + 32];
                    if l2 > 15 {
                        l2 -= 16;
                        qh[j] |= m2;
                    }
                    ql[offset + j] = l1 | (l2 << 4);
                }
                m1 <<= 2;
                m2 <<= 2;
            }
        }

        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        for (block, y) in group_for_dequantization(xs, ys)? {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let ql = &block.qs;
            let qh = &block.qh;
            let mut is = 0;
            let mut u1 = 1;
            let mut u2 = 2;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let ql = &ql[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u1 != 0 { 16 } else { 1 };
                    y[ys_index] = d1 * ((ql & 0xF) + to_add) as f32 - m1;
                    ys_index += 1;
                }
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u2 != 0 { 16 } else { 1 };
                    y[ys_index] = d2 * ((ql >> 4) + to_add) as f32 - m2;
                    ys_index += 1;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ6K {
    const DTYPE: GgmlDType = GgmlDType::Q6K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q6k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q6k_q8k(n, xs, ys);

        if n % QK_K != 0 {
            crate::bail!("vec_dot_q6k_q8k: {n} is not divisible by {QK_K}")
        }

        let mut aux8 = [0i8; QK_K];
        let mut aux16 = [0i16; 8];
        let mut sums = [0f32; 8];
        let mut aux32 = [0f32; 8];

        for (x, y) in xs.iter().zip(ys.iter()) {
            let q4 = &x.ql;
            let qh = &x.qh;
            let q8 = &y.qs;
            aux32.fill(0f32);

            for j in (0..QK_K).step_by(128) {
                let aux8 = &mut aux8[j..];
                let q4 = &q4[j / 2..];
                let qh = &qh[j / 4..];
                for l in 0..32 {
                    aux8[l] = (((q4[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 32] =
                        (((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 64] = (((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 96] =
                        (((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
                }
            }

            for (j, &scale) in x.scales.iter().enumerate() {
                let scale = scale as f32;
                let q8 = &q8[16 * j..];
                let aux8 = &aux8[16 * j..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
                let q8 = &q8[8..];
                let aux8 = &aux8[8..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
            }

            let d = x.d.to_f32() * y.d;
            for (sum, &a) in sums.iter_mut().zip(aux32.iter()) {
                *sum += a * d;
            }
        }
        Ok(sums.iter().sum())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() * Self::BLCK_SIZE {
            crate::bail!(
                "quantize_row_q6k: size mismatch {} {} {}",
                xs.len(),
                ys.len(),
                Self::BLCK_SIZE
            )
        }
        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut x = xs.as_ptr();
        let l = l.as_mut_ptr();
        unsafe {
            for y in ys.iter_mut() {
                let mut max_scale = 0f32;
                let mut max_abs_scale = 0f32;
                for (ib, scale_) in scales.iter_mut().enumerate() {
                    let scale = make_qx_quants(16, 32, x.add(16 * ib), l.add(16 * ib), 1);
                    *scale_ = scale;
                    let abs_scale = scale.abs();
                    if abs_scale > max_abs_scale {
                        max_abs_scale = abs_scale;
                        max_scale = scale
                    }
                }

                let iscale = -128f32 / max_scale;
                y.d = f16::from_f32(1.0 / iscale);

                for (y_scale, scale) in y.scales.iter_mut().zip(scales.iter()) {
                    *y_scale = nearest_int(iscale * scale).min(127) as i8
                }

                for (j, &y_scale) in y.scales.iter().enumerate() {
                    let d = y.d.to_f32() * y_scale as f32;
                    if d == 0. {
                        continue;
                    }
                    for ii in 0..16 {
                        let ll = nearest_int(*x.add(16 * j + ii) / d).clamp(-32, 31);
                        *l.add(16 * j + ii) = (ll + 32) as i8
                    }
                }

                let mut ql = y.ql.as_mut_ptr();
                let mut qh = y.qh.as_mut_ptr();

                for j in (0..QK_K).step_by(128) {
                    for l_idx in 0..32 {
                        let q1 = *l.add(j + l_idx) & 0xF;
                        let q2 = *l.add(j + l_idx + 32) & 0xF;
                        let q3 = *l.add(j + l_idx + 64) & 0xF;
                        let q4 = *l.add(j + l_idx + 96) & 0xF;
                        *ql.add(l_idx) = (q1 | (q3 << 4)) as u8;
                        *ql.add(l_idx + 32) = (q2 | (q4 << 4)) as u8;
                        *qh.add(l_idx) = ((*l.add(j + l_idx) >> 4)
                            | ((*l.add(j + l_idx + 32) >> 4) << 2)
                            | ((*l.add(j + l_idx + 64) >> 4) << 4)
                            | ((*l.add(j + l_idx + 96) >> 4) << 6))
                            as u8;
                    }
                    ql = ql.add(64);
                    qh = qh.add(32);
                }

                x = x.add(QK_K)
            }
        }
        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L1067
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("dequantize_row_q6k: {k} is not divisible by {QK_K}")
        }
        for (idx_x, x) in xs.iter().enumerate() {
            let d = x.d.to_f32();
            let ql = &x.ql;
            let qh = &x.qh;
            let sc = &x.scales;
            for n in (0..QK_K).step_by(128) {
                let idx = n / 128;
                let ys = &mut ys[idx_x * QK_K + n..];
                let sc = &sc[8 * idx..];
                let ql = &ql[64 * idx..];
                let qh = &qh[32 * idx..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    ys[l] = d * sc[is] as f32 * q1 as f32;
                    ys[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                    ys[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                    ys[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                }
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ8K {
    const DTYPE: GgmlDType = GgmlDType::Q8K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        unreachable!()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        let k = xs.len();
        if k % QK_K != 0 {
            crate::bail!("quantize_row_q8k: {k} is not divisible by {QK_K}")
        }
        for (i, y) in ys.iter_mut().enumerate() {
            let mut max = 0f32;
            let mut amax = 0f32;
            let xs = &xs[i * QK_K..(i + 1) * QK_K];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            if amax == 0f32 {
                y.d = 0f32;
                y.qs.fill(0)
            } else {
                let iscale = -128f32 / max;
                for (j, q) in y.qs.iter_mut().enumerate() {
                    // ggml uses nearest_int with bit magic here, maybe we want the same
                    // but we would have to test and benchmark it.
                    let v = (iscale * xs[j]).round();
                    *q = v.min(127.) as i8
                }
                for j in 0..QK_K / 16 {
                    let mut sum = 0i32;
                    for ii in 0..16 {
                        sum += y.qs[j * 16 + ii] as i32
                    }
                    y.bsums[j] = sum as i16
                }
                y.d = 1.0 / iscale
            }
        }
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("dequantize_row_q8k: {k} is not divisible by {QK_K}")
        }
        for (i, x) in xs.iter().enumerate() {
            for (j, &q) in x.qs.iter().enumerate() {
                ys[i * QK_K + j] = x.d * q as f32
            }
        }
        Ok(())
    }
}

// https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L10605
pub fn matmul<T: GgmlType>(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()> {
    let (m, k, n) = mkn;
    if m * k != lhs.len() {
        crate::bail!("unexpected lhs length {} {mkn:?}", lhs.len());
    }

    let k_in_lhs_blocks = (k + T::BLCK_SIZE - 1) / T::BLCK_SIZE;
    let k_in_rhs_blocks = (k + T::VecDotType::BLCK_SIZE - 1) / T::VecDotType::BLCK_SIZE;
    // TODO: Do not make this copy if the DotType is f32.
    // TODO: Pre-allocate this.
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_lhs_blocks];
    for row_idx in 0..m {
        let lhs_b = &mut lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
        T::VecDotType::from_float(lhs, lhs_b)?
    }
    let lhs_b = lhs_b.as_slice();

    for row_idx in 0..m {
        let lhs_row = &lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let dst_row = &mut dst[row_idx * n..(row_idx + 1) * n];

        let result: Result<Vec<_>> = dst_row
            .into_par_iter()
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .map(|(col_idx, dst)| {
                let rhs_col = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
                T::vec_dot(k, rhs_col, lhs_row).map(|value| *dst = value)
            })
            .collect();

        result?;
    }
    Ok(())
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;
    type VecDotType = f32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f32(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }
}

impl GgmlType for f16 {
    const DTYPE: GgmlDType = GgmlDType::F16;
    const BLCK_SIZE: usize = 1;
    type VecDotType = f16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = f16::from_f32(*x)
        }
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = x.to_f32()
        }
        Ok(())
    }
}
