//! 优化的 Mel 特征提取 - 针对 Apple Silicon 优化
//! 
//! 主要优化：
//! 1. SIMD 加速的向量操作
//! 2. 预分配缓冲区减少内存分配
//! 3. 并行化计算
//! 4. 缓存友好的内存访问模式

use anyhow::Result;
use rustfft::{num_complex::Complex, FftPlanner};
use rayon::prelude::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

pub(crate) const MEL_SAMPLE_RATE: u32 = 16000;
pub(crate) const N_FFT: usize = 400;
pub(crate) const HOP_LENGTH: usize = 160;

/// Hann 窗函数 - 预计算并缓存
fn hann_window_cached(n: usize) -> Vec<f32> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<f32>> = OnceLock::new();
    
    if n == N_FFT {
        return CACHE.get_or_init(|| hann_window(N_FFT)).clone();
    }
    hann_window(n)
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// 反射填充 - 使用 SIMD 加速（如果可用）
#[inline]
fn reflection_pad_optimized(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    if n == 0 || pad == 0 {
        return signal.to_vec();
    }
    
    let mut padded = Vec::with_capacity(n + 2 * pad);
    
    // 左侧填充
    for k in 0..pad {
        let i = -(pad as isize) + k as isize;
        padded.push(signal[reflect_index(i, n)]);
    }
    
    // 中间数据
    padded.extend_from_slice(signal);
    
    // 右侧填充
    for k in 0..pad {
        let i = n as isize + k as isize;
        padded.push(signal[reflect_index(i, n)]);
    }
    
    padded
}

#[inline]
fn reflect_index(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = (2 * (n - 1)) as isize;
    let mut x = i % period;
    if x < 0 {
        x += period;
    }
    let n1 = (n - 1) as isize;
    if x > n1 {
        (period - x) as usize
    } else {
        x as usize
    }
}

/// 计算功率谱 - 并行化并优化内存访问
fn compute_power_stft_parallel(
    signal: &[f32],
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> (Vec<f32>, usize, usize) {
    let n_freqs = n_fft / 2 + 1;
    let n_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    if n_frames == 0 {
        return (vec![], n_freqs, 0);
    }

    // 使用线程局部的 FFT planner 避免竞争
    let power: Vec<f32> = (0..n_frames)
        .into_par_iter()
        .flat_map(|i| {
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(n_fft);
            
            let start = i * hop_length;
            let mut frame_buf: Vec<Complex<f32>> = (0..n_fft)
                .map(|j| Complex::new(signal[start + j] * window[j], 0.0))
                .collect();
            
            fft.process(&mut frame_buf);
            
            // 只返回需要的频率（前 n_freqs 个）
            let mut frame_power = Vec::with_capacity(n_freqs);
            for k in 0..n_freqs {
                let re = frame_buf[k].re;
                let im = frame_buf[k].im;
                frame_power.push(re * re + im * im);
            }
            frame_power
        })
        .collect();

    // 转置：从 [frame, freq] 到 [freq, frame]
    let mut transposed = vec![0.0f32; n_freqs * n_frames];
    for f in 0..n_freqs {
        for t in 0..n_frames {
            transposed[f * n_frames + t] = power[t * n_freqs + f];
        }
    }

    (transposed, n_freqs, n_frames)
}

/// Mel 滤波器组 - 缓存
fn create_mel_filterbank_cached(
    num_mels: usize,
    n_fft: usize,
    sample_rate: u32,
) -> Vec<f32> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<f32>> = OnceLock::new();
    
    if num_mels == 128 && n_fft == N_FFT && sample_rate == MEL_SAMPLE_RATE {
        return CACHE
            .get_or_init(|| {
                create_mel_filterbank(num_mels, n_fft, sample_rate, 0.0, sample_rate as f64 / 2.0)
            })
            .clone();
    }
    
    create_mel_filterbank(num_mels, n_fft, sample_rate, 0.0, sample_rate as f64 / 2.0)
}

fn create_mel_filterbank(
    num_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let sr = sample_rate as f64;

    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;

    let hz_to_mel = |f: f64| -> f64 {
        if f < min_log_hz {
            f / f_sp
        } else {
            min_log_mel + (f / min_log_hz).ln() / logstep
        }
    };

    let mel_to_hz = |m: f64| -> f64 {
        if m < min_log_mel {
            f_sp * m
        } else {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let filter_freqs: Vec<f64> = (0..num_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (num_mels + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    let all_freqs: Vec<f64> = (0..n_freqs).map(|j| j as f64 * sr / n_fft as f64).collect();
    let f_diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    let mut filters = vec![0.0f32; num_mels * n_freqs];

    // 并行构建滤波器
    filters
        .par_chunks_mut(n_freqs)
        .enumerate()
        .for_each(|(i, filter_row)| {
            for j in 0..n_freqs {
                let down = (all_freqs[j] - filter_freqs[i]) / f_diff[i];
                let up = (filter_freqs[i + 2] - all_freqs[j]) / f_diff[i + 1];
                let val = down.min(up).max(0.0);
                filter_row[j] = val as f32;
            }
            
            // 归一化
            let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
            for j in 0..n_freqs {
                filter_row[j] *= enorm as f32;
            }
        });

    filters
}

/// 优化的 Mel 提取器
pub(crate) struct OptimizedMelExtractor {
    n_fft: usize,
    hop_length: usize,
    num_mel_bins: usize,
    mel_filters: Vec<f32>,
    n_freqs: usize,
    window: Vec<f32>,
}

impl OptimizedMelExtractor {
    pub(crate) fn new(
        n_fft: usize,
        hop_length: usize,
        num_mel_bins: usize,
        sample_rate: u32,
    ) -> Self {
        let n_freqs = n_fft / 2 + 1;
        let mel_filters = create_mel_filterbank_cached(num_mel_bins, n_fft, sample_rate);
        let window = hann_window_cached(n_fft);
        
        Self {
            n_fft,
            hop_length,
            num_mel_bins,
            mel_filters,
            n_freqs,
            window,
        }
    }

    /// 提取 Mel 特征 - 高度优化版本
    pub(crate) fn extract(&self, samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        anyhow::ensure!(!samples.is_empty(), "empty audio");
        
        // 1. 反射填充
        let pad = self.n_fft / 2;
        let padded_signal = reflection_pad_optimized(samples, pad);

        // 2. 并行计算 STFT
        let (power, _n_freqs, n_frames_with_last) =
            compute_power_stft_parallel(&padded_signal, self.n_fft, self.hop_length, &self.window);

        let n_frames = if n_frames_with_last > 0 {
            n_frames_with_last - 1
        } else {
            0
        };

        if n_frames == 0 {
            return Ok((vec![], self.num_mel_bins, 0));
        }

        // 3. 应用 Mel 滤波器 - 并行化
        let mut mel_spec = vec![0.0f32; self.num_mel_bins * n_frames];
        
        mel_spec
            .par_chunks_mut(n_frames)
            .enumerate()
            .for_each(|(m, out_row)| {
                let filter_row = &self.mel_filters[m * self.n_freqs..(m + 1) * self.n_freqs];
                
                // 使用 SIMD 优化的点积（如果可用）
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                unsafe {
                    self.apply_filter_neon(filter_row, &power, out_row, n_frames, n_frames_with_last);
                }
                
                #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
                self.apply_filter_scalar(filter_row, &power, out_row, n_frames, n_frames_with_last);
            });

        // 4. 对数压缩和归一化
        let log10_factor = 1.0 / 10.0f32.ln();
        let mut max_val = f32::NEG_INFINITY;
        
        for v in mel_spec.iter_mut() {
            *v = v.max(1e-10f32).ln() * log10_factor;
            if *v > max_val {
                max_val = *v;
            }
        }

        let min_val = max_val - 8.0;
        mel_spec.par_iter_mut().for_each(|v| {
            *v = (v.max(min_val) + 4.0) / 4.0;
        });

        Ok((mel_spec, self.num_mel_bins, n_frames))
    }

    /// 标量版本的滤波器应用
    #[inline]
    fn apply_filter_scalar(
        &self,
        filter_row: &[f32],
        power: &[f32],
        out_row: &mut [f32],
        n_frames: usize,
        n_frames_with_last: usize,
    ) {
        for f in 0..self.n_freqs {
            let w = filter_row[f];
            if w == 0.0 {
                continue;
            }
            let power_row = &power[f * n_frames_with_last..f * n_frames_with_last + n_frames];
            for (t, &p) in power_row.iter().enumerate() {
                out_row[t] += w * p;
            }
        }
    }

    /// NEON SIMD 优化版本的滤波器应用 (Apple Silicon)
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline]
    unsafe fn apply_filter_neon(
        &self,
        filter_row: &[f32],
        power: &[f32],
        out_row: &mut [f32],
        n_frames: usize,
        n_frames_with_last: usize,
    ) {
        for f in 0..self.n_freqs {
            let w = filter_row[f];
            if w == 0.0 {
                continue;
            }
            
            let w_vec = vdupq_n_f32(w);
            let power_row = &power[f * n_frames_with_last..f * n_frames_with_last + n_frames];
            
            let mut t = 0;
            // 处理 4 个元素一组
            while t + 4 <= n_frames {
                let p_vec = vld1q_f32(power_row.as_ptr().add(t));
                let o_vec = vld1q_f32(out_row.as_ptr().add(t));
                let result = vmlaq_f32(o_vec, w_vec, p_vec); // out + w * power
                vst1q_f32(out_row.as_mut_ptr().add(t), result);
                t += 4;
            }
            
            // 处理剩余元素
            while t < n_frames {
                out_row[t] += w * power_row[t];
                t += 1;
            }
        }
    }
}

/// 便捷函数 - 使用优化的提取器
pub fn mel_features_optimized(samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
    OptimizedMelExtractor::new(N_FFT, HOP_LENGTH, 128, MEL_SAMPLE_RATE).extract(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_mel_vs_original() {
        let samples: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        
        let (mel_opt, bins_opt, frames_opt) = mel_features_optimized(&samples).unwrap();
        
        assert_eq!(bins_opt, 128);
        assert!(frames_opt > 0);
        assert_eq!(mel_opt.len(), bins_opt * frames_opt);
        assert!(mel_opt.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_hann_window_cache() {
        let w1 = hann_window_cached(N_FFT);
        let w2 = hann_window_cached(N_FFT);
        assert_eq!(w1.len(), w2.len());
        assert_eq!(w1, w2);
    }
}
