//! Host-side carrier for the run's 16-bit storage format.
//!
//! f16 and bf16 are the same width and the same packing — two per `u32` for
//! activations, eight per `vec4<u32>` for weights — so a value "in the storage
//! format" can travel as its raw 16 bits and be interpreted on demand.  That is
//! what [`H16`] is, and it is what lets one upload path serve both formats
//! without knowing which one is running: the narrowing functions in
//! [`crate::weights`] round through it, the buffers hold the same bits either
//! way, and only [`H16::to_f32`] has to know the format at the far end.
//!
//! The format itself is [`crate::shaders::half`], fixed once per process.

use crate::shaders::{self, Half};

/// One element of the 16-bit storage format, as the bits that reach the device.
///
/// The bits are only meaningful together with the format they were written in,
/// which is why this type does not implement `Add` or `PartialOrd`: a value has
/// to be widened ([`H16::to_f32`]) before arithmetic means anything, and the
/// widening is where the format is read.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct H16(u16);

impl H16 {
    pub const ZERO: H16 = H16(0);

    /// `v` rounded into the run's format.
    ///
    /// Round-to-nearest-even, the rule `torch.Tensor.to(torch.bfloat16)` uses:
    /// a bf16 tensor narrowing to bf16 is therefore an identity, and f32 weights
    /// land on the same bits the reference's own cast produces.
    #[inline]
    pub fn from_f32(v: f32) -> H16 {
        H16::from_f32_in(v, shaders::half())
    }

    /// [`H16::from_f32`] in `half`'s format, for a caller holding the choice
    /// explicitly rather than taking it from the process-wide setting.
    #[inline]
    pub fn from_f32_in(v: f32, half: Half) -> H16 {
        H16(match half {
            Half::F16 => half::f16::from_f32(v).to_bits(),
            // The mirror of the WGSL `rne_bf16`, which is pinned against the
            // `half` crate by `shaders::bf16_round_tests`.
            Half::Bf16 => shaders::rne_bf16_bits(v),
        })
    }

    /// The value the device reads back out of these bits.
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.to_f32_in(shaders::half())
    }

    /// [`H16::to_f32`] in `half`'s format.
    #[inline]
    pub fn to_f32_in(self, half: Half) -> f32 {
        match half {
            Half::F16 => half::f16::from_bits(self.0).to_f32(),
            Half::Bf16 => f32::from_bits((self.0 as u32) << 16),
        }
    }

    /// These bits as a `Half` would spell them, with no conversion — the read
    /// side of a buffer whose format the caller already knows.
    #[inline]
    pub fn from_bits(bits: u16) -> H16 {
        H16(bits)
    }

    #[inline]
    pub fn to_bits(self) -> u16 {
        self.0
    }

    #[inline]
    pub fn from_le_bytes(b: [u8; 2]) -> H16 {
        H16(u16::from_le_bytes(b))
    }

    #[inline]
    pub fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value either format can hold is an f32 exactly, so narrowing and
    /// widening has to come back to the same bits — that exactness is what keeps
    /// the readback path format-blind.
    #[test]
    fn both_formats_round_trip_through_f32() {
        for &x in &[0.0f32, 1.0, -1.0, 0.1, 3.14159, 1e-5, 6.5e4, -2.5e-3] {
            for h in Half::ALL {
                let bits = H16::from_f32_in(x, h);
                assert_eq!(
                    H16::from_f32_in(bits.to_f32_in(h), h),
                    bits,
                    "{} {x}",
                    h.name()
                );
            }
        }
    }

    /// The bf16 arm rounds to nearest even rather than truncating: 0.1f32 is
    /// 0x3DCCCCCD, and its discarded half is past the midpoint, so the keeper
    /// becomes 0x3DCD.
    #[test]
    fn bf16_rounds_rather_than_truncates() {
        let bf = H16::from_f32_in(0.1, Half::Bf16);
        assert_eq!(bf.to_bits(), 0x3DCD);
        assert_ne!(bf.to_bits(), (0.1f32.to_bits() >> 16) as u16);
        // The bare `to_f32` reads the process-wide setting, which is f16 here;
        // everything in this test names its format explicitly.
        assert_eq!(bf.to_f32_in(Half::Bf16), f32::from_bits(0x3DCD_0000));
        // f16 keeps 10 mantissa bits, so it is a different set of values.
        let f = H16::from_f32_in(0.1, Half::F16);
        assert_ne!(f.to_bits(), bf.to_bits());
        assert_eq!(f.to_f32_in(Half::F16), half::f16::from_f32(0.1).to_f32());
    }

    /// The process-wide setting is what `from_f32` / `to_f32` read: a caller that
    /// never names a format gets whatever *this run* fixed before its first
    /// pipeline existed.
    ///
    /// Asserted against `shaders::half()` rather than against a literal, because
    /// the setting is process-wide and the test binary runs its tests in
    /// parallel — pinning the value here would make this test fail whenever
    /// another test legitimately changes the format.  The default itself is
    /// pinned in `shaders::the_format_defaults_to_f16`, where it belongs.
    #[test]
    fn the_bare_constructors_follow_the_setting() {
        let running = shaders::half();
        assert_eq!(H16::from_f32(0.1), H16::from_f32_in(0.1, running));
        assert_eq!(
            H16::from_f32(0.1).to_f32(),
            H16::from_f32_in(0.1, running).to_f32()
        );
        // The two formats really are different bits for this value, so the
        // assertions above are not vacuous.
        assert_ne!(
            H16::from_f32_in(0.1, Half::F16),
            H16::from_f32_in(0.1, Half::Bf16)
        );
    }

    #[test]
    fn zero_is_zero_in_both() {
        for h in Half::ALL {
            assert_eq!(H16::ZERO.to_f32_in(h), 0.0);
            assert_eq!(H16::from_f32_in(-0.0, h).to_f32_in(h), 0.0);
        }
    }
}
