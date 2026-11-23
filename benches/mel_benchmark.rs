use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use qwen3_aligner_wgpu::mel::{mel_features, MEL_SAMPLE_RATE};

// 生成测试音频数据
fn generate_test_audio(duration_secs: usize) -> Vec<f32> {
    let sample_rate = MEL_SAMPLE_RATE as usize;
    let n_samples = sample_rate * duration_secs;
    
    // 生成包含多个频率的复杂信号
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let sig1 = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3;
            let sig2 = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.2;
            let sig3 = (2.0 * std::f32::consts::PI * 1320.0 * t).sin() * 0.15;
            sig1 + sig2 + sig3
        })
        .collect()
}

fn bench_mel_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_extraction");
    
    for duration in [1, 5, 10, 30].iter() {
        let samples = generate_test_audio(*duration);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}s", duration)),
            &samples,
            |b, samples| {
                b.iter(|| {
                    mel_features(black_box(samples)).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_mel_different_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_various_lengths");
    
    // 测试不同长度的音频片段
    for n_samples in [4000, 8000, 16000, 32000, 48000].iter() {
        let samples: Vec<f32> = (0..*n_samples)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", n_samples)),
            &samples,
            |b, samples| {
                b.iter(|| {
                    mel_features(black_box(samples)).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_mel_extraction, bench_mel_different_lengths);
criterion_main!(benches);
