//! Timestamp repair and the word/timestamp pairing.
//!
//! The head emits one class per 80 ms bucket per `<timestamp>` token, and the
//! classes do not have to increase.  Repair is part of the output contract:
//!
//! 1. Take the longest increasing subsequence (O(n^2) DP) and call those
//!    timestamps "normal".  Ties are broken by *earliest index*: `max` and
//!    `index` both scan forward, and `data[prev] <= data[current]` is
//!    non-strict, so a plateau folds into the chain.
//! 2. Every maximal run of non-normal positions is replaced:
//!    * length <= 2 — snap each position to whichever surrounding normal value
//!      is nearer, by *index distance* (`pos - (start-1) <= (end - pos)`), not by
//!      value;
//!    * longer — linearly interpolate `left + (right-left)/(n+1) * k` in f64,
//!      with `k = 1..n`.
//! 3. Truncate to integer with `int()`, which is **toward zero**, not `round`;
//!    the interpolation branch makes non-multiples of 80 ms real values on the
//!    output path.
//!
//! Left/right neighbours are read out of `result`, not `data` — identical here
//! because they are by construction at normal positions and only non-normal
//! positions are ever written, but the port keeps the reference's read.

/// One aligned word.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignItem {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// `_fix_timestamps`: force a monotonically non-decreasing sequence.
pub fn fix_timestamps(data: &[i64]) -> Vec<i64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let mut dp = vec![1usize; n];
    let mut parent = vec![-1i64; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j as i64;
            }
        }
    }

    let max_length = *dp.iter().max().expect("non-empty");
    let max_idx = dp.iter().position(|&v| v == max_length).expect("max exists");

    let mut is_normal = vec![false; n];
    let mut idx = max_idx as i64;
    while idx != -1 {
        is_normal[idx as usize] = true;
        idx = parent[idx as usize];
    }

    let mut result = data.to_vec();
    let mut block_start = 0usize;
    while block_start < n {
        if is_normal[block_start] {
            block_start += 1;
            continue;
        }
        let mut block_end = block_start;
        while block_end < n && !is_normal[block_end] {
            block_end += 1;
        }
        let anomaly_count = block_end - block_start;

        let left_val = (0..block_start)
            .rev()
            .find(|&k| is_normal[k])
            .map(|k| result[k]);
        let right_val = (block_end..n).find(|&k| is_normal[k]).map(|k| result[k]);

        // The LIS is never empty, so at least one neighbour always exists; the
        // (None, None) combination is unreachable and left as a no-op.
        if left_val.is_some() || right_val.is_some() {
            if anomaly_count <= 2 {
                for pos in block_start..block_end {
                    result[pos] = match (left_val, right_val) {
                        (Some(l), Some(r)) => {
                            if pos - (block_start - 1) <= block_end - pos {
                                l
                            } else {
                                r
                            }
                        }
                        (Some(l), None) => l,
                        (None, Some(r)) => r,
                        (None, None) => unreachable!(),
                    };
                }
            } else if let (Some(l), Some(r)) = (left_val, right_val) {
                let step = (r - l) as f64 / (anomaly_count + 1) as f64;
                for pos in block_start..block_end {
                    result[pos] = (l as f64 + step * (pos - block_start + 1) as f64) as i64;
                }
            } else if let Some(l) = left_val {
                result[block_start..block_end].fill(l);
            } else if let Some(r) = right_val {
                result[block_start..block_end].fill(r);
            }
        }

        block_start = block_end;
    }

    result
}

/// Pair the repaired millisecond list with the words: `(2i, 2i+1)` per word.
///
/// The reference wraps each value in `round(ms / 1000.0, 3)`.  That is a no-op
/// here and is deliberately not reproduced: `fix_timestamps` has already
/// truncated to an integer number of milliseconds, and the nearest double to
/// `ms / 1000` is by definition the nearest double to a multiple of 10^-3, so
/// rounding it to 3 decimals returns the same double.  What has to match is the
/// 3-decimal *formatting*, which happens at print time in both languages.
pub fn pair_words(
    words: &[String],
    fixed_ms: &[i64],
) -> anyhow::Result<Vec<AlignItem>> {
    if fixed_ms.len() != words.len() * 2 {
        anyhow::bail!(
            "expected {} timestamps for {} words, got {}",
            words.len() * 2,
            words.len(),
            fixed_ms.len()
        );
    }
    Ok(words
        .iter()
        .enumerate()
        .map(|(i, w)| AlignItem {
            text: w.clone(),
            start_time: fixed_ms[i * 2] as f64 / 1000.0,
            end_time: fixed_ms[i * 2 + 1] as f64 / 1000.0,
        })
        .collect())
}

/// The full decode step: repair, then pair.  `raw_ms` is `argmax * 80`.
pub fn decode_timestamps(words: &[String], raw_ms: &[i64]) -> anyhow::Result<Vec<AlignItem>> {
    pair_words(words, &fix_timestamps(raw_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn already_monotonic_is_untouched() {
        let data = [0i64, 80, 160, 240];
        assert_eq!(fix_timestamps(&data), data.to_vec());
    }

    #[test]
    fn plateau_is_not_an_anomaly() {
        // `<=` is non-strict, so equal values stay on the LIS.
        let data = [5i64, 5, 5, 5];
        assert_eq!(fix_timestamps(&data), data.to_vec());
    }

    #[test]
    fn short_block_snaps_to_the_nearer_neighbour_by_index_distance() {
        // LIS = {0, 1, 4} (0, 100, 200), so indices 2..4 are the block.
        // pos 2: 2 - (2-1) = 1 <= 4 - 2 = 2 -> left   (100)
        // pos 3: 3 - (2-1) = 2 <= 4 - 3 = 1 -> false  -> right (200)
        // The rule is index distance, not value distance: 5 is far closer to 100
        // in value than to 200, and still snaps right.
        let data = [0i64, 100, 5, 3, 200];
        assert_eq!(fix_timestamps(&data), vec![0, 100, 100, 200, 200]);
    }

    #[test]
    fn long_block_interpolates() {
        // LIS = {0, 1, 5} (0, 5, 40); the block is indices 2..5 (length 3).
        // step = (40 - 5) / (3 + 1) = 8.75 -> 13.75, 22.5, 31.25
        let data = [0i64, 5, 4, 3, 2, 40];
        assert_eq!(fix_timestamps(&data), vec![0, 5, 13, 22, 31, 40]);
    }

    #[test]
    fn interpolation_truncates_toward_zero() {
        // step = (101 - 4) / 4 = 24.25 -> 28.25, 52.5, 76.75 -> 28, 52, 76
        let data = [0i64, 4, 3, 2, 1, 101];
        assert_eq!(fix_timestamps(&data), vec![0, 4, 28, 52, 76, 101]);
    }

    #[test]
    fn trailing_block_without_a_right_neighbour_holds_the_left_value() {
        // LIS = {0, 1, 2} (0, 80, 160); the tail 3..6 has nothing normal after it.
        let data = [0i64, 80, 160, 5, 4, 3];
        assert_eq!(fix_timestamps(&data), vec![0, 80, 160, 160, 160, 160]);
    }

    #[test]
    fn leading_block_without_a_left_neighbour_takes_the_right_value() {
        // LIS = {3, 4, 5, 6} (10, 20, 30, 40); indices 0..3 are a length-3
        // leading block with no normal predecessor.
        let data = [50i64, 40, 30, 10, 20, 30, 40];
        assert_eq!(fix_timestamps(&data), vec![10, 10, 10, 10, 20, 30, 40]);
    }

    #[test]
    fn single_leading_outlier_snaps_right() {
        // LIS = {1, 2, 3}; only index 0 is out of order.
        let data = [50i64, 10, 20, 30];
        assert_eq!(fix_timestamps(&data), vec![10, 10, 20, 30]);
    }

    #[test]
    fn output_is_monotonic_after_repair() {
        let data = [500i64, 10, 400, 20, 30, 900, 5, 5, 5, 950];
        let fixed = fix_timestamps(&data);
        assert!(fixed.windows(2).all(|w| w[0] <= w[1]), "{fixed:?}");
    }

    #[test]
    fn pairing_follows_the_documented_layout() {
        let words: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let items = decode_timestamps(&words, &[2000, 2080, 2080, 2240]).unwrap();
        assert_eq!(
            items[0],
            AlignItem { text: "a".into(), start_time: 2.0, end_time: 2.08 }
        );
        assert_eq!(
            items[1],
            AlignItem { text: "b".into(), start_time: 2.08, end_time: 2.24 }
        );
    }

    #[test]
    fn pairing_rejects_a_length_mismatch() {
        let words: Vec<String> = vec!["a".into()];
        assert!(decode_timestamps(&words, &[0, 1, 2]).is_err());
    }
}
