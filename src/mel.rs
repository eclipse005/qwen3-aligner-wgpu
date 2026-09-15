use anyhow::Result;
use rustfft::{num_complex::Complex, FftPlanner};

pub(crate) const MEL_SAMPLE_RATE: u32 = 16000;
pub(crate) const N_FFT: usize = 400;
pub(crate) const HOP_LENGTH: usize = 160;

fn hann_window(n: usize) -> Vec<f32> {
    // Match torch.hann_window(n, periodic=True) used by Qwen3ASRFeatureExtractor.
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// numpy/torch `mode="reflect"` (edge not repeated). `n==0` is empty.
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

fn reflection_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    if n == 0 || pad == 0 {
        return signal.to_vec();
    }
    let mut padded = Vec::with_capacity(n + 2 * pad);
    for k in 0..pad {
        let i = -(pad as isize) + k as isize;
        padded.push(signal[reflect_index(i, n)]);
    }
    padded.extend_from_slice(signal);
    for k in 0..pad {
        let i = n as isize + k as isize;
        padded.push(signal[reflect_index(i, n)]);
    }
    padded
}

fn compute_power_stft(
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

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut power = vec![0.0f32; n_freqs * n_frames];
    let mut frame_buf = vec![Complex::new(0.0f32, 0.0f32); n_fft];

    for i in 0..n_frames {
        let start = i * hop_length;
        for j in 0..n_fft {
            frame_buf[j] = Complex::new(signal[start + j] * window[j], 0.0);
        }
        fft.process(&mut frame_buf);
        for k in 0..n_freqs {
            let re = frame_buf[k].re;
            let im = frame_buf[k].im;
            power[k * n_frames + i] = re * re + im * im;
        }
    }

    (power, n_freqs, n_frames)
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

    for j in 0..n_freqs {
        for i in 0..num_mels {
            let down = (all_freqs[j] - filter_freqs[i]) / f_diff[i];
            let up = (filter_freqs[i + 2] - all_freqs[j]) / f_diff[i + 1];
            let val = down.min(up).max(0.0);
            filters[i * n_freqs + j] = val as f32;
        }
    }

    for i in 0..num_mels {
        let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
        for j in 0..n_freqs {
            filters[i * n_freqs + j] *= enorm as f32;
        }
    }

    filters
}

pub(crate) struct MelExtractor {
    n_fft: usize,
    hop_length: usize,
    num_mel_bins: usize,
    mel_filters: Vec<f32>,
    n_freqs: usize,
}

impl MelExtractor {
    pub(crate) fn new(
        n_fft: usize,
        hop_length: usize,
        num_mel_bins: usize,
        sample_rate: u32,
    ) -> Self {
        let n_freqs = n_fft / 2 + 1;
        let mel_filters = create_mel_filterbank(
            num_mel_bins,
            n_fft,
            sample_rate,
            0.0,
            sample_rate as f64 / 2.0,
        );
        Self {
            n_fft,
            hop_length,
            num_mel_bins,
            mel_filters,
            n_freqs,
        }
    }

    pub(crate) fn extract(&self, samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        anyhow::ensure!(!samples.is_empty(), "empty audio");
        // torch.stft(center=True): reflect-pad n_fft/2, then drop the extra frame.
        let pad = self.n_fft / 2;
        let padded_signal = reflection_pad(samples, pad);

        let window = hann_window(self.n_fft);

        let (power, _n_freqs, n_frames_with_last) =
            compute_power_stft(&padded_signal, self.n_fft, self.hop_length, &window);

        let n_frames = if n_frames_with_last > 0 {
            n_frames_with_last - 1
        } else {
            0
        };

        let mut mel_spec = vec![0.0f32; self.num_mel_bins * n_frames];
        for m in 0..self.num_mel_bins {
            let filter_row = &self.mel_filters[m * self.n_freqs..(m + 1) * self.n_freqs];
            let out_row = &mut mel_spec[m * n_frames..(m + 1) * n_frames];
            for f in 0..self.n_freqs {
                let w = filter_row[f];
                if w == 0.0 { continue; }
                let power_row = &power[f * n_frames_with_last..f * n_frames_with_last + n_frames];
                for (t, &p) in power_row.iter().enumerate() {
                    out_row[t] += w * p;
                }
            }
        }

        let log10_factor = 1.0 / 10.0f32.ln();
        let mut max_val = f32::NEG_INFINITY;
        for v in mel_spec.iter_mut() {
            *v = v.max(1e-10f32).ln() * log10_factor;
            if *v > max_val { max_val = *v; }
        }

        let min_val = max_val - 8.0;
        for v in mel_spec.iter_mut() {
            *v = (v.max(min_val) + 4.0) / 4.0;
        }

        Ok((mel_spec, self.num_mel_bins, n_frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_periodic() {
        let w = hann_window(4);
        assert_eq!(w.len(), 4);
        assert!(w[0].abs() < 1e-6);
        assert!((w[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reflection_pad_basic() {
        let signal = vec![1.0f32, 2.0, 3.0];
        let padded = reflection_pad(&signal, 1);
        assert_eq!(padded, vec![2.0f32, 1.0, 2.0, 3.0, 2.0]);
        assert_eq!(
            reflection_pad(&signal, 4),
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
        );
        assert!(reflection_pad(&[], 3).is_empty());
    }

    #[test]
    fn test_mel_filterbank_shape_and_nonneg() {
        let num_mels = 128;
        let n_fft = 400;
        let sr = 16000u32;
        let n_freqs = n_fft / 2 + 1;
        let filters = create_mel_filterbank(num_mels, n_fft, sr, 0.0, sr as f64 / 2.0);
        assert_eq!(filters.len(), num_mels * n_freqs);
        assert!(filters.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_mel_extractor_silent_signal() {
        let samples = vec![0.0f32; 16000];
        let extractor = MelExtractor::new(400, 160, 128, 16000);
        let (mel, n_mels, n_frames) = extractor.extract(&samples).unwrap();
        assert_eq!(n_mels, 128);
        assert!(n_frames > 0);
        assert_eq!(mel.len(), n_mels * n_frames);
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_resample_44100_to_16000_length() {
        let n = 44100usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let y = resample_soxr(&x, 44100, 16000).unwrap();
        let expected = (n as f64 * 16000.0 / 44100.0).ceil() as usize;
        assert_eq!(y.len(), expected);
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(y.iter().any(|v| *v != 0.0));
    }

    /// The session's window slices must reproduce the whole-clip frames; the
    /// only difference the extractor can introduce is its per-call log-mel max
    /// normalization.  This locks that property down: every value moves by
    /// `0 ..= -(M_whole - M_slice)/4` (large values are untouched, deep silence
    /// clamps to the same floor on both sides).
    #[test]
    fn slice_extraction_matches_whole_clip_modulo_normalization() {
        let wav = r"D:\qwen3-asr-rs\tests\fixtures\180s_zh.wav";
        if !std::path::Path::new(wav).exists() {
            return;
        }
        let samples = load_audio_wav(wav, MEL_SAMPLE_RATE).unwrap();
        let ex = MelExtractor::new(N_FFT, HOP_LENGTH, 128, MEL_SAMPLE_RATE);
        let (mel_w, n_mels, frames_w) = ex.extract(&samples).unwrap();
        let m_whole = mel_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 4.0 - 4.0;

        // Window 1 of the session's slicing: `2` hops of lead-in, `2` dropped frames.
        let (win, drop) = (800usize, 2usize);
        let f0 = win;
        let lo = f0 * HOP_LENGTH - drop * HOP_LENGTH;
        let hi = (f0 + win - 1) * HOP_LENGTH + N_FFT / 2;
        let (mel_s, _n, frames_s) = ex.extract(&samples[lo..hi]).unwrap();
        let m_slice = mel_s.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 4.0 - 4.0;
        let shift = (m_whole - m_slice) / 4.0;

        let (mut dmin, mut dmax) = (f32::INFINITY, f32::NEG_INFINITY);
        for m in 0..n_mels {
            for j in 0..win {
                let d = mel_s[m * frames_s + drop + j] - mel_w[m * frames_w + f0 + j];
                dmin = dmin.min(d);
                dmax = dmax.max(d);
            }
        }
        println!(
            "M_whole {m_whole:.6} M_slice {m_slice:.6} shift {shift:.6} delta [{dmin:.6}, {dmax:.6}]"
        );
        assert!(shift > 0.0, "fixture should have a window max below the whole-clip max");
        assert!(dmax.abs() < 1e-6, "slice values must never exceed the whole-clip ones: {dmax}");
        assert!((dmin + shift).abs() < 1e-5, "delta floor {dmin} != -shift {}", -shift);
    }

    #[test]
    fn resample_180s_en_vs_python_dump() {
        let wav = r"D:\qwen3-asr-rs\tests\fixtures\180s_en.wav";
        let py_path = r"D:\qwen3-asr-wgpu\align_dump\py_180s_en\wave16k.f32";
        if !std::path::Path::new(wav).exists() || !std::path::Path::new(py_path).exists() {
            return;
        }
        let rust = load_audio_wav(wav, 16000).unwrap();
        let raw = std::fs::read(py_path).unwrap();
        let py: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n = rust.len().min(py.len());
        let mut dot = 0.0f64;
        let mut nr = 0.0f64;
        let mut np_ = 0.0f64;
        let mut maxabs = 0.0f32;
        for i in 0..n {
            let a = rust[i] as f64;
            let b = py[i] as f64;
            dot += a * b;
            nr += a * a;
            np_ += b * b;
            maxabs = maxabs.max((rust[i] - py[i]).abs());
        }
        let corr = dot / (nr.sqrt() * np_.sqrt());
        println!(
            "rust={} py={} dlen={} corr={corr:.8} maxabs={maxabs:.6e}",
            rust.len(),
            py.len(),
            rust.len() as i64 - py.len() as i64
        );
        assert!(
            corr > 0.9999,
            "soxr vs python wave corr {corr} maxabs {maxabs}"
        );
    }

    #[test]
    fn load_and_mel_180s_zh_vs_python_dump() {
        let wav = r"D:\qwen3-asr-rs\tests\fixtures\180s_zh.wav";
        let py_wave = r"D:\qwen3-asr-wgpu\align_dump\py_180s_zh\wave16k.f32";
        let py_mel = r"D:\qwen3-asr-wgpu\align_dump\py_180s_zh\mel.f32";
        if !std::path::Path::new(wav).exists() || !std::path::Path::new(py_wave).exists() {
            return;
        }
        let rust = load_audio_wav(wav, 16000).unwrap();
        let py: Vec<f32> = std::fs::read(py_wave)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!("wave rust={} py={}", rust.len(), py.len());
        let n = rust.len().min(py.len());
        let mut maxabs = 0.0f32;
        let mut ndiff = 0usize;
        for i in 0..n {
            let d = (rust[i] - py[i]).abs();
            if d > 0.0 {
                ndiff += 1;
            }
            maxabs = maxabs.max(d);
        }
        println!("wave maxabs={maxabs:.6e} ndiff={ndiff}");

        let extractor = MelExtractor::new(N_FFT, HOP_LENGTH, 128, MEL_SAMPLE_RATE);
        let (mel, n_mels, n_frames) = extractor.extract(&rust).unwrap();
        let pmel: Vec<f32> = std::fs::read(py_mel)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!(
            "mel rust={}x{} ({}) py={}",
            n_mels,
            n_frames,
            mel.len(),
            pmel.len()
        );
        let m = mel.len().min(pmel.len());
        let mut mmax = 0.0f32;
        let mut mrms = 0.0f64;
        let mut mdiff = 0usize;
        for i in 0..m {
            let d = (mel[i] - pmel[i]).abs();
            if d > 1e-5 {
                mdiff += 1;
            }
            mmax = mmax.max(d);
            mrms += (d as f64) * (d as f64);
        }
        mrms = (mrms / m as f64).sqrt();
        println!("mel maxabs={mmax:.6e} rms={mrms:.6e} ndiff>1e-5={mdiff}");
    }
}

pub fn load_audio_wav(path: impl AsRef<std::path::Path>, target_sr: u32) -> anyhow::Result<Vec<f32>> {
    load_audio_wav_impl(path.as_ref(), target_sr)
}

fn load_audio_wav_impl(path: &std::path::Path, target_sr: u32) -> anyhow::Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;
    let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;

    // A data chunk shorter than the header promises is not fatal: libsndfile and
    // ffmpeg play what is there, and one FLEURS test wav ships that way (its
    // `data` chunk claims 151492 bytes, the file holds 147398).  Keep the
    // samples that could be read rather than failing the whole clip.
    let mut truncated = false;
    let mut samples_f32: Vec<f32> = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for s in reader.into_samples::<f32>() {
                match s {
                    Ok(v) => samples_f32.push(v),
                    Err(_) => {
                        truncated = true;
                        break;
                    }
                }
            }
        }
        hound::SampleFormat::Int => {
            for s in reader.into_samples::<i32>() {
                match s {
                    Ok(v) => samples_f32.push(v as f32 / max_val),
                    Err(_) => {
                        truncated = true;
                        break;
                    }
                }
            }
        }
    }
    anyhow::ensure!(!samples_f32.is_empty(), "WAV read error: no samples in {}", path.display());
    if truncated {
        eprintln!(
            "warning: {} is truncated (header promises more samples), using {}",
            path.display(),
            samples_f32.len()
        );
    }

    let mono: Vec<f32> = if channels == 1 {
        samples_f32
    } else {
        samples_f32
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if sr == target_sr {
        return Ok(mono);
    }
    resample_soxr(&mono, sr, target_sr)
}

/// Match Transformers `load_audio` → librosa (`soxr_hq`).
///
/// librosa.resample runs soxr HQ then `fix_length` to `ceil(n * target / orig)`.
/// `soxr_oneshot` defaults to LQ — quality must be passed explicitly.
fn resample_soxr(mono: &[f32], sr: u32, target_sr: u32) -> anyhow::Result<Vec<f32>> {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_uint, c_ulong, c_void};

    #[repr(C)]
    struct SoxrQualitySpec {
        precision: f64,
        phase_response: f64,
        passband_end: f64,
        stopband_begin: f64,
        e: *mut c_void,
        flags: c_ulong,
    }

    #[repr(C)]
    struct SoxrRuntimeSpec {
        log2_min_dft_size: c_uint,
        log2_large_dft_size: c_uint,
        coef_size_kbytes: c_uint,
        num_threads: c_uint,
        e: *mut c_void,
        flags: c_ulong,
    }

    type SoxrT = *mut c_void;
    type SoxrErrorT = *const c_char;

    const SOXR_HQ: c_ulong = 4; // SOXR_20_BITQ

    unsafe extern "C" {
        fn soxr_quality_spec(recipe: c_ulong, flags: c_ulong) -> SoxrQualitySpec;
        fn soxr_runtime_spec(num_threads: c_uint) -> SoxrRuntimeSpec;
        fn soxr_create(
            input_rate: f64,
            output_rate: f64,
            num_channels: c_uint,
            err: *mut SoxrErrorT,
            io_spec: *const c_void,
            quality_spec: *const SoxrQualitySpec,
            runtime_spec: *const SoxrRuntimeSpec,
        ) -> SoxrT;
        fn soxr_process(
            resampler: SoxrT,
            in_: *const c_void,
            ilen: usize,
            idone: *mut usize,
            out: *mut c_void,
            olen: usize,
            odone: *mut usize,
        ) -> SoxrErrorT;
        fn soxr_delete(resampler: SoxrT);
    }

    fn soxr_err(err: SoxrErrorT) -> anyhow::Result<()> {
        if err.is_null() {
            return Ok(());
        }
        let msg = unsafe { CStr::from_ptr(err) }.to_string_lossy();
        anyhow::bail!("soxr: {msg}")
    }

    // librosa.resample(..., fix=True) uses ceil, not trunc/round.
    let expected = (mono.len() as f64 * target_sr as f64 / sr as f64).ceil() as usize;
    let q = unsafe { soxr_quality_spec(SOXR_HQ, 0) };
    // One thread on purpose: soxr's OpenMP path only splits work across
    // *channels* (`num_channels > 1` in `soxr.c`), and this is mono, so a thread
    // pool buys nothing.  Measured: 1 vs 8 threads, same 1.3 s and byte-identical
    // output for a 3-minute 44.1 kHz clip.
    let rt = unsafe { soxr_runtime_spec(1) };
    let mut err: SoxrErrorT = std::ptr::null();
    let soxr = unsafe {
        soxr_create(
            sr as f64,
            target_sr as f64,
            1,
            &mut err,
            std::ptr::null(),
            &q,
            &rt,
        )
    };
    soxr_err(err)?;
    anyhow::ensure!(!soxr.is_null(), "soxr_create returned null");

    let mut out = vec![0.0f32; expected + 8192];
    let mut in_off = 0usize;
    let mut written = 0usize;
    let result = (|| -> anyhow::Result<Vec<f32>> {
        while in_off < mono.len() {
            let mut idone = 0usize;
            let mut odone = 0usize;
            let remain_out = out.len() - written;
            anyhow::ensure!(remain_out > 0, "soxr output overflow");
            let err = unsafe {
                soxr_process(
                    soxr,
                    mono[in_off..].as_ptr().cast(),
                    mono.len() - in_off,
                    &mut idone,
                    out[written..].as_mut_ptr().cast(),
                    remain_out,
                    &mut odone,
                )
            };
            soxr_err(err)?;
            if idone == 0 && odone == 0 {
                anyhow::bail!("soxr made no progress");
            }
            in_off += idone;
            written += odone;
        }
        loop {
            let remain_out = out.len() - written;
            if remain_out == 0 {
                break;
            }
            let mut idone = 0usize;
            let mut odone = 0usize;
            let err = unsafe {
                soxr_process(
                    soxr,
                    std::ptr::null(),
                    0,
                    &mut idone,
                    out[written..].as_mut_ptr().cast(),
                    remain_out,
                    &mut odone,
                )
            };
            soxr_err(err)?;
            if odone == 0 {
                break;
            }
            written += odone;
        }
        out.truncate(written);
        if out.len() > expected {
            out.truncate(expected);
        } else if out.len() < expected {
            out.resize(expected, 0.0);
        }
        Ok(out)
    })();
    unsafe { soxr_delete(soxr) };
    result
}
