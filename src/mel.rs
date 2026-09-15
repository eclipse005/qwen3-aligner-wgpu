use anyhow::Result;
use rustfft::{num_complex::Complex, FftPlanner};

pub(crate) const MEL_SAMPLE_RATE: u32 = 16000;
pub(crate) const N_FFT: usize = 400;
pub(crate) const HOP_LENGTH: usize = 160;

/// Log-mel for the aligner: 16 kHz, `n_fft` 400, hop 160, 128 slaney bins,
/// `log10` clamped to `max - 8` then `(x + 4) / 4` (`librosa`-compatible).
///
/// Returns `(mel, num_mel_bins, valid_frames)`, row-major `[bins, frames]`.  The
/// caller still has to right-pad the time axis to a multiple of `n_window * 2`
/// with `0.0` (see `align_input::padded_mel_frames`) — that padding is a
/// processor-level step, not part of the STFT.
pub fn mel_features(samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
    MelExtractor::new(N_FFT, HOP_LENGTH, 128, MEL_SAMPLE_RATE).extract(samples)
}

fn hann_window(n: usize) -> Vec<f32> {
    // Periodic Hann window: 0.5 * (1 - cos(2*pi*i/n)).
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// Reflection index with the edge not repeated (`mode="reflect"`). `n <= 1`
/// returns 0.
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
        // Centered STFT: reflect-pad n_fft/2, then drop the trailing frame.
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

}

pub fn load_audio_wav(path: impl AsRef<std::path::Path>, target_sr: u32) -> anyhow::Result<Vec<f32>> {
    load_audio_wav_impl(path.as_ref(), target_sr)
}

/// Read a plain 16-bit PCM wav straight out of the data chunk, decoding samples
/// in place as `i16 as f32 / 32768.0`.
///
/// Returns `None` for anything that is not mono-able 16-bit PCM — float formats,
/// other bit depths, extensible headers — and the caller falls back to `hound`.
/// The chunk walk skips unknown chunks and honours the odd-size pad byte.
fn read_pcm16_fast(path: &std::path::Path) -> Option<(Vec<f32>, u32, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " if size >= 16 => {
                let f = u16::from_le_bytes(bytes[body_start..body_start + 2].try_into().ok()?);
                let ch = u16::from_le_bytes(bytes[body_start + 2..body_start + 4].try_into().ok()?);
                let rate = u32::from_le_bytes(bytes[body_start + 4..body_start + 8].try_into().ok()?);
                let bits = u16::from_le_bytes(bytes[body_start + 14..body_start + 16].try_into().ok()?);
                fmt = Some((f, ch, rate, bits));
            }
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        // Chunks are word-aligned: an odd size carries a pad byte.
        pos = body_start + size + (size & 1);
        if pos <= body_start {
            break;
        }
    }
    let (format, channels, rate, bits) = fmt?;
    if format != 1 || bits != 16 || channels == 0 || channels > 8 {
        return None;
    }
    let data = data?;
    let frames = data.len() / (2 * channels as usize);
    let mut out = Vec::with_capacity(frames);
    for frame in data[..frames * 2 * channels as usize].chunks_exact(2 * channels as usize) {
        if channels == 1 {
            out.push(i16::from_le_bytes([frame[0], frame[1]]) as f32 / 32768.0);
        } else {
            let mut acc = 0.0f32;
            for c in frame.chunks_exact(2) {
                acc += i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0;
            }
            out.push(acc / channels as f32);
        }
    }
    Some((out, rate, channels as usize))
}

fn load_audio_wav_impl(path: &std::path::Path, target_sr: u32) -> anyhow::Result<Vec<f32>> {
    if let Some((mono, sr, _ch)) = read_pcm16_fast(path) {
        return finish_audio(mono, sr, target_sr);
    }
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;
    let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;

    // A data chunk shorter than the header promises is not fatal: keep the
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

    finish_audio(mono, sr, target_sr)
}

/// Resample if needed.  Shared by the fast and the `hound` path so the two can
/// only ever differ in how the samples were decoded, never in what follows.
fn finish_audio(mono: Vec<f32>, sr: u32, target_sr: u32) -> anyhow::Result<Vec<f32>> {
    if sr == target_sr {
        return Ok(mono);
    }
    resample_soxr(&mono, sr, target_sr)
}

#[cfg(test)]
mod fast_reader_tests {
    use super::*;

    /// The fast PCM16 reader must agree with `hound` **bit for bit**: a one-ULP
    /// difference would propagate through soxr into the mel and out as a moved
    /// timestamp.
    #[test]
    fn fast_pcm16_equals_hound_for_every_fixture() {
        let dir = crate::paths::fixtures_dir();
        if !dir.is_dir() {
            return;
        }
        let mut checked = 0usize;
        for clip in crate::paths::CLIPS {
            let path = dir.join(format!("{clip}.wav"));
            if !path.is_file() {
                continue;
            }
            let fast = read_pcm16_fast(&path).map(|(m, _sr, _ch)| m);
            assert!(fast.is_some(), "{clip}: fast path declined a plain PCM16 wav");
            let fast = fast.unwrap();

            // The hound path, verbatim, so the comparison is against the code
            // path the fast reader replaces rather than against a restatement.
            let reader = hound::WavReader::open(&path).unwrap();
            let spec = reader.spec();
            assert_eq!(spec.sample_format, hound::SampleFormat::Int);
            assert_eq!(spec.bits_per_sample, 16);
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            let ch = spec.channels as usize;
            let raw: Vec<f32> = reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect();
            let slow: Vec<f32> = if ch == 1 {
                raw
            } else {
                raw.chunks(ch)
                    .map(|c| c.iter().sum::<f32>() / ch as f32)
                    .collect()
            };

            assert_eq!(fast.len(), slow.len(), "{clip}: sample count");
            for (i, (a, b)) in fast.iter().zip(&slow).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{clip}: sample {i} differs: fast={a} hound={b}"
                );
            }
            checked += 1;
        }
        assert!(checked > 0, "no fixtures found under {}", dir.display());
    }
}

/// Resample with soxr HQ, `librosa`-style: soxr then `fix_length` to
/// `ceil(n * target / orig)`.  `soxr_oneshot` defaults to LQ, so the quality
/// must be passed explicitly.
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
    // One thread: soxr's OpenMP path only splits work across *channels*, and this
    // is mono, so a thread pool buys nothing.
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
