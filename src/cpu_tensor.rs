//! Host f32 tensors + GEMM used by the CPU audio encoder.

use gemm::{gemm, Parallelism};
use rayon::prelude::*;

pub struct CpuTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "CpuTensor len mismatch (shape {:?})", shape);
        Self { data, shape }
    }
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

pub struct CpuWeight {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub struct CpuWeightF16 {
    pub data: Vec<half::f16>,
    pub rows: usize,
    pub cols: usize,
}

impl CpuWeightF16 {
    pub fn to_f32(&self) -> CpuWeight {
        let data: Vec<f32> = self.data.iter().map(|v| v.to_f32()).collect();
        CpuWeight {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

pub fn linear(x: &CpuTensor, w: &CpuWeight) -> CpuTensor {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let n = w.rows;
    assert_eq!(k, w.cols, "linear K mismatch: x last={} vs W cols={}", k, w.cols);
    let mut out_shape = x.shape.clone();
    out_shape[nd - 1] = n;
    if m == 1 {
        let out = linear_gemv(&x.data, w);
        return CpuTensor::new(out, out_shape);
    }
    let mut out = vec![0.0f32; m * n];
    gemm_row_major(&mut out, &x.data, w, m, 0.0);
    CpuTensor::new(out, out_shape)
}

fn linear_gemv(x: &[f32], w: &CpuWeight) -> Vec<f32> {
    let n = w.rows;
    let k = w.cols;
    let mut out = vec![0.0f32; n];
    let chunk = (n / (rayon::current_num_threads() * 4)).max(64).min(2048);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slab)| {
        let row0 = ci * chunk;
        for (offset, o) in slab.iter_mut().enumerate() {
            let row = row0 + offset;
            let w_row = &w.data[row * k..(row + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += x[j] * w_row[j];
            }
            *o = acc;
        }
    });
    out
}

fn gemm_row_major(out: &mut [f32], x: &[f32], w: &CpuWeight, m: usize, beta: f32) {
    let n = w.rows;
    let k = w.cols;
    unsafe {
        gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            beta != 0.0,
            x.as_ptr(),
            1,
            k as isize,
            w.data.as_ptr(),
            k as isize,
            1,
            beta,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }
}
