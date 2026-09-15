//! MRoPE (multi-axis RoPE) cos/sin tables — self-contained.
//!
//! The text decoder rotates `head_dim`-wide rows where the *first half* of the
//! frequency index space is split across three position axes (temporal / height /
//! width for multimodal input, all three equal for pure text) according to
//! `mrope_section`, and the second half duplicates the first.
//!
//! Everything is computed in f64 and rounded to f32 once, exactly like the
//! reference implementation, so the resulting table is bit-identical.

/// Expand `section` (one quota per axis) into a per-frequency axis index.
///
/// `interleaved` picks the Qwen2-VL layout (`0,1,2,0,1,2,…` while quotas last)
/// versus the blocked layout (all of axis 0, then all of axis 1, …).
pub fn build_dim_map(section: &[usize], half: usize, interleaved: bool) -> Vec<usize> {
    if interleaved {
        let nd = section.len();
        let mut m = Vec::with_capacity(half);
        let mut used = vec![0usize; nd];
        while m.len() < half {
            let before = m.len();
            for d in 0..nd {
                if m.len() >= half {
                    break;
                }
                if used[d] < section[d] {
                    m.push(d);
                    used[d] += 1;
                }
            }
            if m.len() == before {
                break;
            }
        }
        m
    } else {
        let mut m = Vec::with_capacity(half);
        for (d, &sz) in section.iter().enumerate() {
            for _ in 0..sz {
                if m.len() >= half {
                    break;
                }
                m.push(d);
            }
        }
        while m.len() < half {
            m.push(section.len() - 1);
        }
        m
    }
}

/// `(cos, sin)`, each `[seq_len, head_dim]` row-major.
///
/// `pos[a][t]` is the position of token `t` on axis `a`.
pub fn compute_mrope_cos_sin(
    pos: &[Vec<i64>; 3],
    head_dim: usize,
    rope_theta: f64,
    section: &[usize],
    interleaved: bool,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let seq_len = pos[0].len();
    let inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();
    let dm = build_dim_map(section, half, interleaved);

    let mut cv = vec![0.0f32; seq_len * head_dim];
    let mut sv = vec![0.0f32; seq_len * head_dim];
    for t in 0..seq_len {
        for j in 0..half {
            let angle = pos[dm[j]][t] as f64 * inv[j];
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            cv[t * head_dim + j] = c;
            sv[t * head_dim + j] = s;
            cv[t * head_dim + j + half] = c;
            sv[t * head_dim + j + half] = s;
        }
    }
    (cv, sv)
}

/// Text-only helper: all three axes share the same position sequence `0..n`.
pub fn text_positions(n: usize) -> [Vec<i64>; 3] {
    let p: Vec<i64> = (0..n as i64).collect();
    [p.clone(), p.clone(), p]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_map_respects_quotas() {
        let m = build_dim_map(&[24, 20, 20], 64, true);
        assert_eq!(m.len(), 64);
        assert_eq!(m.iter().filter(|&&d| d == 0).count(), 24);
        assert_eq!(m.iter().filter(|&&d| d == 1).count(), 20);
        assert_eq!(m.iter().filter(|&&d| d == 2).count(), 20);
        assert_eq!(&m[..6], &[0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn blocked_map_is_grouped() {
        let m = build_dim_map(&[2, 1], 4, false);
        assert_eq!(m, vec![0, 0, 1, 1]);
    }

    #[test]
    fn text_positions_duplicate_halves() {
        let (c, _s) = compute_mrope_cos_sin(&text_positions(3), 8, 1e6, &[2, 1, 1], true);
        assert_eq!(c.len(), 3 * 8);
        // second half duplicates the first
        assert_eq!(c[0..4], c[4..8]);
    }
}
