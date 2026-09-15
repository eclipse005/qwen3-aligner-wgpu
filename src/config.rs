#![allow(dead_code)]

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AsrConfig {
    pub thinker_config: ThinkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfig {
    pub audio_config: AudioEncoderConfig,
    pub text_config: TextDecoderConfig,
    #[serde(default = "default_audio_start_token_id")]
    pub audio_start_token_id: i64,
    #[serde(default = "default_audio_end_token_id")]
    pub audio_end_token_id: i64,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: i64,
}

fn default_audio_start_token_id() -> i64 { 151669 }
fn default_audio_end_token_id() -> i64 { 151670 }
fn default_audio_token_id() -> i64 { 151676 }

#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    /// Conv-stem output width = `d_model` of the `conv_out` projection input.
    /// Present in `config.json` (`downsample_hidden_size`, 480 for both sizes).
    #[serde(default = "default_downsample_hidden_size")]
    pub downsample_hidden_size: usize,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,
    #[serde(default = "default_output_dim")]
    pub output_dim: usize,
}

fn default_d_model() -> usize { 896 }
fn default_encoder_layers() -> usize { 18 }
fn default_encoder_attention_heads() -> usize { 14 }
fn default_encoder_ffn_dim() -> usize { 3584 }
fn default_num_mel_bins() -> usize { 128 }
fn default_downsample_hidden_size() -> usize { 480 }
fn default_max_source_positions() -> usize { 1500 }
fn default_n_window() -> usize { 50 }
fn default_n_window_infer() -> usize { 800 }
fn default_conv_chunksize() -> usize { 500 }
fn default_output_dim() -> usize { 1024 }

#[derive(Debug, Clone, Deserialize)]
pub struct TextDecoderConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    /// Transformers-native `-hf` checkpoints store theta under `rope_parameters`.
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

fn default_vocab_size() -> usize { 151936 }
fn default_hidden_size() -> usize { 1024 }
fn default_intermediate_size() -> usize { 3072 }
fn default_num_hidden_layers() -> usize { 28 }
fn default_num_attention_heads() -> usize { 16 }
fn default_num_key_value_heads() -> usize { 8 }
fn default_head_dim() -> usize { 128 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f64 { 1_000_000.0 }

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub rope_type: String,
    #[serde(default = "default_mrope_section")]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub interleaved: bool,
    #[serde(default)]
    pub mrope_interleaved: bool,
}

fn default_mrope_section() -> Vec<usize> { vec![24, 20, 20] }

/// On-disk `config.json`: either the original `thinker_config` layout or the
/// Transformers-native `-hf` layout (`audio_config` + `text_config` at top level).
#[derive(Debug, Deserialize)]
struct AsrConfigFile {
    #[serde(default)]
    thinker_config: Option<ThinkerConfig>,
    #[serde(default)]
    audio_config: Option<AudioEncoderConfig>,
    #[serde(default)]
    text_config: Option<TextDecoderConfig>,
    #[serde(default)]
    audio_token_id: Option<i64>,
    #[serde(default)]
    audio_start_token_id: Option<i64>,
    #[serde(default)]
    audio_end_token_id: Option<i64>,
}

impl AsrConfig {
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json_str(&content).map_err(|e| anyhow::anyhow!("{}: {}", path.display(), e))
    }

    fn from_json_str(content: &str) -> anyhow::Result<Self> {
        let raw: AsrConfigFile = serde_json::from_str(content)?;
        let mut thinker = if let Some(tc) = raw.thinker_config {
            tc
        } else {
            let audio_config = raw.audio_config.ok_or_else(|| {
                anyhow::anyhow!("missing thinker_config and audio_config")
            })?;
            let text_config = raw.text_config.ok_or_else(|| {
                anyhow::anyhow!("missing thinker_config and text_config")
            })?;
            ThinkerConfig {
                audio_config,
                text_config,
                audio_start_token_id: raw.audio_start_token_id.unwrap_or_else(default_audio_start_token_id),
                audio_end_token_id: raw.audio_end_token_id.unwrap_or_else(default_audio_end_token_id),
                audio_token_id: raw.audio_token_id.unwrap_or_else(default_audio_token_id),
            }
        };
        if let Some(rp) = thinker.text_config.rope_parameters.as_ref() {
            if let Some(theta) = rp.rope_theta {
                thinker.text_config.rope_theta = theta;
            }
        }
        Ok(AsrConfig { thinker_config: thinker })
    }
}

impl TextDecoderConfig {
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_section.clone())
            .unwrap_or_else(default_mrope_section)
    }

    pub fn mrope_interleaved(&self) -> bool {
        self.rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_interleaved || rs.interleaved)
            .unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_base_config() -> TextDecoderConfig {
        TextDecoderConfig {
            vocab_size: 151936,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            rope_parameters: None,
        }
    }

    #[test]
    fn test_mrope_section_default_when_no_rope_scaling() {
        let cfg = make_base_config();
        assert_eq!(cfg.mrope_section(), vec![24, 20, 20]);
    }

    #[test]
    fn test_mrope_section_custom() {
        let mut cfg = make_base_config();
        cfg.rope_scaling = Some(RopeScaling {
            rope_type: "mrope".to_string(),
            mrope_section: vec![4, 4, 8],
            interleaved: false,
            mrope_interleaved: false,
        });
        assert_eq!(cfg.mrope_section(), vec![4, 4, 8]);
    }

    #[test]
    fn test_mrope_interleaved_default_true_when_no_rope_scaling() {
        let cfg = make_base_config();
        assert!(cfg.mrope_interleaved());
    }

    #[test]
    fn test_mrope_interleaved_via_mrope_interleaved_flag() {
        let mut cfg = make_base_config();
        cfg.rope_scaling = Some(RopeScaling {
            rope_type: "mrope".to_string(),
            mrope_section: vec![24, 20, 20],
            interleaved: false,
            mrope_interleaved: true,
        });
        assert!(cfg.mrope_interleaved());
    }

    #[test]
    fn test_mrope_interleaved_via_interleaved_flag() {
        let mut cfg = make_base_config();
        cfg.rope_scaling = Some(RopeScaling {
            rope_type: "mrope".to_string(),
            mrope_section: vec![24, 20, 20],
            interleaved: true,
            mrope_interleaved: false,
        });
        assert!(cfg.mrope_interleaved());
    }

    #[test]
    fn test_parse_hf_native_config_layout() {
        let json = r#"{
            "audio_config": {
                "d_model": 896,
                "encoder_layers": 18,
                "encoder_attention_heads": 14,
                "encoder_ffn_dim": 3584,
                "num_mel_bins": 128,
                "n_window": 50,
                "n_window_infer": 800,
                "conv_chunksize": 500,
                "output_dim": 1024
            },
            "audio_token_id": 151676,
            "text_config": {
                "vocab_size": 151936,
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "rope_parameters": { "rope_theta": 1000000, "rope_type": "default" }
            }
        }"#;
        let raw: AsrConfigFile = serde_json::from_str(json).unwrap();
        assert!(raw.thinker_config.is_none());
        assert_eq!(raw.audio_config.unwrap().encoder_layers, 18);
        let theta = raw.text_config.unwrap().rope_parameters.unwrap().rope_theta.unwrap();
        assert_eq!(theta, 1_000_000.0);
        let cfg = AsrConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.thinker_config.audio_config.encoder_layers, 18);
        assert_eq!(cfg.thinker_config.text_config.rope_theta, 1_000_000.0);
        assert_eq!(cfg.thinker_config.audio_token_id, 151676);
        assert_eq!(cfg.thinker_config.audio_start_token_id, 151669);
        assert_eq!(cfg.thinker_config.audio_end_token_id, 151670);
    }

    #[test]
    fn test_load_shipped_legacy_and_hf_configs() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let legacy = root.join("models/Qwen3-ASR-0.6B/config.json");
        if legacy.is_file() {
            let cfg = AsrConfig::from_file(&legacy).unwrap();
            assert_eq!(cfg.thinker_config.audio_config.encoder_layers, 18);
            assert_eq!(cfg.thinker_config.text_config.num_hidden_layers, 28);
        }
        let hf = std::path::Path::new(r"D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf\config.json");
        if hf.is_file() {
            let cfg = AsrConfig::from_file(hf).unwrap();
            assert_eq!(cfg.thinker_config.audio_config.encoder_layers, 18);
            assert_eq!(cfg.thinker_config.text_config.hidden_size, 1024);
            assert_eq!(cfg.thinker_config.text_config.rope_theta, 1_000_000.0);
            assert_eq!(cfg.thinker_config.audio_token_id, 151676);
        }
        let hf17 = std::path::Path::new(r"D:\Qwen3-ASR\models\Qwen3-ASR-1.7B-hf\config.json");
        if hf17.is_file() {
            let cfg = AsrConfig::from_file(hf17).unwrap();
            assert_eq!(cfg.thinker_config.audio_config.encoder_layers, 24);
            assert_eq!(cfg.thinker_config.text_config.hidden_size, 2048);
            assert_eq!(cfg.thinker_config.audio_config.output_dim, 2048);
        }
    }

    #[test]
    fn test_mrope_interleaved_both_false() {
        let mut cfg = make_base_config();
        cfg.rope_scaling = Some(RopeScaling {
            rope_type: "mrope".to_string(),
            mrope_section: vec![24, 20, 20],
            interleaved: false,
            mrope_interleaved: false,
        });
        assert!(!cfg.mrope_interleaved());
    }
}
