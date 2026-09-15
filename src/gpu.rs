//! Device / queue wrapper and buffer plumbing.
//!
//! Everything the engine needs from wgpu lives here so the rest of the crate can
//! stay in terms of buffers + dispatches.  Two conventions matter for the whole
//! engine and are enforced by the constructors below:
//!
//! * storage buffers are always allocated 16-byte padded, so a `array<vec4<u32>>`
//!   view over an f16 payload never runs off the end;
//! * every activation tensor is stored the way CUDA stores it — f16, indexed as
//!   `__half2` words (2 halves per `u32`).  Keeping the byte layout identical to
//!   the CUDA backend is what lets the two store bit-comparable values.

use anyhow::{bail, Context, Result};

/// A wgpu device plus the queue, adapter info and negotiated limits.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
    /// Compiled-pipeline cache (see [`pipeline_cache_path`]).  `None` when the
    /// adapter does not offer the feature.
    pub pipeline_cache: Option<wgpu::PipelineCache>,
    /// Where that cache is persisted, if it is.
    pub pipeline_cache_path: Option<std::path::PathBuf>,
}

/// Which device to run on.
///
/// One binary, several devices is the whole point of the port, so the choice is
/// explicit and enumerable rather than "whatever wgpu hands back": a caller can
/// run on the integrated GPU while the discrete one is busy, pin a backend for
/// an A/B, or address the same machine's adapters by index.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    /// The default: the best GPU, on the runtime this engine is tuned for
    /// (Vulkan first, then Metal, then D3D12, then GL — see [`rank`]).
    #[default]
    Auto,
    /// A compute runtime and which device of it: `Runtime { api: Vulkan, index: 1 }`
    /// for `vulkan:1`.  This is the axis users mean by "backend" — the same
    /// shape as ONNX Runtime's execution providers and llama.cpp's
    /// `CUDA0`/`Vulkan0`/`CPU` device names.
    ///
    /// Note what is *not* in here: the vendor.  Which vendor a card is belongs in
    /// the listing, not in the choice — an NVIDIA card can be driven through
    /// Vulkan, D3D12 or DML, and those are different code paths with different
    /// numerics and different speeds.
    Runtime { api: wgpu::Backend, index: usize },
    /// The host implementation: the CPU audio tower plus
    /// [`crate::cpu_decoder::CpuTextDecoder`].  Needs no adapter at all, and is
    /// also what [`Self::Auto`] falls back to when no GPU can be created.
    Cpu,
    /// Debug/addressing view: index into [`list_devices`] (one entry per
    /// *(device, runtime)* pair) — for disambiguating what the listing printed.
    Index(usize),
    /// Last resort: a case-insensitive substring of the adapter name.
    Name(String),
}

/// The compute runtimes this engine can run on, in the order it tries them.
///
/// `cpu` is our own implementation (see [`DeviceSelector::Cpu`]).  Note there is
/// no `cuda` and no `dml`: this crate has no CUDA backend (that lives in the
/// sibling CUDA port), and wgpu drives Windows through D3D12 **compute**, not
/// DirectML.
pub const RUNTIMES: &[(&str, wgpu::Backend)] = &[
    ("vulkan", wgpu::Backend::Vulkan),
    ("metal", wgpu::Backend::Metal),
    ("dx12", wgpu::Backend::Dx12),
    ("d3d12", wgpu::Backend::Dx12),
    ("gl", wgpu::Backend::Gl),
    ("webgpu", wgpu::Backend::BrowserWebGpu),
];

impl DeviceSelector {
    /// Parse a CLI-style spec.
    ///
    /// * `auto` — the default policy
    /// * `cpu` — the CPU implementation (currently refuses, see [`Self::Cpu`])
    /// * `<runtime>[:<index>]` — `vulkan`, `vulkan:1`, `dx12`, `metal:0`, `gl`
    /// * `#<n>` / `<n>` — raw index into [`list_devices`]
    /// * anything else — substring of the adapter name
    pub fn parse(spec: &str) -> Result<Self> {
        let s = spec.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if let Some(rest) = s.strip_prefix('#') {
            return Ok(Self::Index(rest.trim().parse().context("device index")?));
        }
        if let Ok(i) = s.parse::<usize>() {
            return Ok(Self::Index(i));
        }
        // `<runtime>[:<index>]`
        let (name, index) = match s.split_once(':') {
            Some((n, i)) => (n, i.trim().parse().context("runtime device index")?),
            None => (s, 0usize),
        };
        if let Some((_, api)) = RUNTIMES.iter().find(|(n, _)| name.eq_ignore_ascii_case(n)) {
            return Ok(Self::Runtime { api: *api, index });
        }
        Ok(Self::Name(s.to_lowercase()))
    }

    fn backends(&self) -> wgpu::Backends {
        match self {
            Self::Cpu => wgpu::Backends::empty(),
            _ => wgpu::Backends::all(),
        }
    }

    fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
        match self {
            Self::Auto => true,
            Self::Name(n) => info.name.to_lowercase().contains(n),
            Self::Index(_) | Self::Cpu => true,
            Self::Runtime { api, .. } => info.backend == *api,
        }
    }
}

/// Default-selection order: discrete before integrated before virtual/CPU, then
/// by graphics API.  Vulkan first because it is what this engine is tuned and
/// verified against (`docs/PORTING.md`: the same 1070 decodes 54 % faster on
/// Vulkan than on D3D12 here, and one iGPU finished a clip on Vulkan that D3D12
/// did not finish in nine minutes), then Metal for macOS, then D3D12, then GL.
fn rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let class = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        _ => 4,
    };
    let api = match info.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        wgpu::Backend::Gl => 3,
        _ => 4,
    };
    (class, api)
}

/// `#0 NVIDIA … (Vulkan, DiscreteGpu), #1 Intel …` — for "no such device" errors.
fn list_names(adapters: &[wgpu::Adapter]) -> String {
    adapters
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let info = a.get_info();
            format!("#{i} {} ({:?}, {:?})", info.name, info.backend, info.device_type)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// One enumerated adapter — everything you need to *choose* a device, and to
/// know whether it can run this engine at all, without creating one.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
    pub driver: String,
    pub driver_info: String,
    /// `max_storage_buffer_binding_size` — the limit the tiling exists for.
    pub max_binding_bytes: u64,
    pub max_workgroup_storage: u32,
    pub subgroup: bool,
    /// The adapter's promised subgroup width range.  Kernels that fold lane xors
    /// need `32..=32`; anything else must take the shared-memory path.
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    pub timestamps: bool,
    /// Whether this runtime can persist compiled pipelines (`VkPipelineCache` /
    /// `ID3D12PipelineLibrary`) — the fix for D3D12's ~5.5-minute build.
    pub pipeline_cache: bool,
    /// PCI ids — information for the listing, never a selector.
    pub vendor_id: u32,
    pub device_id: u32,
}

impl DeviceInfo {
    /// One line, same shape as [`Gpu::describe`].
    pub fn describe(&self) -> String {
        let sg = if !self.subgroup {
            "none".to_string()
        } else if self.subgroup_min == self.subgroup_max {
            self.subgroup_min.to_string()
        } else {
            format!("{}..{}", self.subgroup_min, self.subgroup_max)
        };
        format!(
            "{} ({:?}, {:?}) | {} {} | wg workgroup storage {} B, binding {} MiB, subgroup {sg}",
            self.name,
            self.backend,
            self.device_type,
            self.driver,
            self.driver_info,
            self.max_workgroup_storage,
            self.max_binding_bytes / (1024 * 1024),
        )
    }

    fn from_adapter(a: &wgpu::Adapter) -> Self {
        let i = a.get_info();
        let l = a.limits();
        let f = a.features();
        Self {
            name: i.name,
            backend: i.backend,
            device_type: i.device_type,
            driver: i.driver,
            driver_info: i.driver_info,
            max_binding_bytes: l.max_storage_buffer_binding_size,
            max_workgroup_storage: l.max_compute_workgroup_storage_size,
            subgroup: f.contains(wgpu::Features::SUBGROUP),
            subgroup_min: i.subgroup_min_size,
            subgroup_max: i.subgroup_max_size,
            timestamps: f.contains(wgpu::Features::TIMESTAMP_QUERY),
            pipeline_cache: f.contains(wgpu::Features::PIPELINE_CACHE),
            vendor_id: i.vendor,
            device_id: i.device,
        }
    }
}

/// Every adapter this instance can see, in wgpu's enumeration order — the order
/// [`DeviceSelector::Index`] indexes into.  Note this is one entry per *(device,
/// graphics API)* pair: the same physical GPU appears once per API it is
/// reachable through.  Use [`list_device_groups`] for the device view.
pub async fn list_devices() -> Vec<DeviceInfo> {
    let instance = wgpu::Instance::default();
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .await
        .iter()
        .map(DeviceInfo::from_adapter)
        .collect()
}

/// One selectable target, named the way [`DeviceSelector::parse`] wants it:
/// `vulkan:0`, `dx12:1`, …  This is the user-facing list — the runtime is the
/// axis, the vendor and device class are *information*, because a card of one
/// vendor is reachable through several runtimes and those are different code
/// paths (see the module docs and `docs/PORTING.md`).
#[derive(Debug, Clone)]
pub struct DeviceTarget {
    /// `<runtime>:<index>`, e.g. `vulkan:1` — feed it back via `--device`.
    pub spec: String,
    pub info: DeviceInfo,
    /// True when [`DeviceSelector::Auto`] would pick this target.
    pub is_default: bool,
}

impl DeviceTarget {
    /// `vulkan:0  NVIDIA P104-100 (NVIDIA, dGPU, driver 572.75) binding 2047 MiB, subgroup 32`
    pub fn describe(&self) -> String {
        let sg = if !self.info.subgroup {
            "no subgroup".to_string()
        } else if self.info.subgroup_min == self.info.subgroup_max {
            format!("subgroup {}", self.info.subgroup_min)
        } else {
            format!("subgroup {}..{}", self.info.subgroup_min, self.info.subgroup_max)
        };
        format!(
            "{}{:<10} {} ({}, {}, driver {}) binding {} MiB, {sg}, pso-cache {}",
            if self.is_default { "* " } else { "  " },
            self.spec,
            self.info.name,
            vendor_label(self.info.vendor_id),
            self.info.device_type_str(),
            self.info.driver,
            self.info.max_binding_bytes / (1024 * 1024),
            if self.info.pipeline_cache { "yes" } else { "no" },
        )
    }
}

impl DeviceInfo {
    fn device_type_str(&self) -> &'static str {
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => "dGPU",
            wgpu::DeviceType::IntegratedGpu => "iGPU",
            wgpu::DeviceType::VirtualGpu => "vGPU",
            wgpu::DeviceType::Cpu => "CPU",
            _ => "other",
        }
    }
}

fn vendor_label(id: u32) -> &'static str {
    match id {
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        0x106B => "Apple",
        0x1414 => "Microsoft",
        _ => "other",
    }
}

/// The user-facing device list: every adapter, named `<runtime>:<index>`, with
/// the runtime's devices ordered discrete-before-integrated (so the index is
/// stable), and the whole list ordered by [`rank`] with the default marked.
pub async fn list_targets() -> Vec<DeviceTarget> {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let mut infos: Vec<DeviceInfo> = adapters.iter().map(DeviceInfo::from_adapter).collect();
    infos.sort_by_key(|d| (rank_of(d), d.name.clone()));

    let mut per_runtime: std::collections::HashMap<wgpu::Backend, usize> =
        std::collections::HashMap::new();
    let mut targets: Vec<DeviceTarget> = Vec::new();
    for info in infos {
        let api = info.backend;
        let n = *per_runtime.entry(api).or_insert(0);
        per_runtime.insert(api, n + 1);
        targets.push(DeviceTarget {
            spec: format!("{}:{n}", runtime_name(api)),
            info,
            is_default: false,
        });
    }
    if let Some(first) = targets.first_mut() {
        first.is_default = true;
    }
    targets
}

/// The name users type for a runtime (and what `warn`/`skip` messages say).
pub fn runtime_name(api: wgpu::Backend) -> &'static str {
    RUNTIMES
        .iter()
        .find(|(_, b)| *b == api)
        .map(|(n, _)| *n)
        .unwrap_or("other")
}

fn rank_of(d: &DeviceInfo) -> (u8, u8) {
    let class = match d.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        _ => 4,
    };
    let api = match d.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        wgpu::Backend::Gl => 3,
        _ => 4,
    };
    (class, api)
}

impl Gpu {
    /// Enumerate adapters and pick one.  `prefer` matches a case-insensitive
    /// substring of the adapter name (e.g. `"nvidia"`, `"intel"`); without it the
    /// first discrete GPU wins, falling back to whatever is available.
    ///
    /// Shorthand for [`Gpu::new_with`] with [`DeviceSelector::Name`] / `Auto`.
    pub async fn new(prefer: Option<&str>) -> Result<Self> {
        let sel = match prefer {
            Some(p) => DeviceSelector::parse(p)?,
            None => DeviceSelector::Auto,
        };
        Self::new_with(sel).await
    }

    /// The explicit form: choose a runtime (`vulkan`, `dx12`, `metal`, `gl`) and
    /// which device of it, or a raw index / name.
    ///
    /// Shorthand: `Gpu::new(Some("vulkan:1"))`.
    pub async fn new_with(selector: DeviceSelector) -> Result<Self> {
        if selector == DeviceSelector::Cpu {
            // `WgpuAsr::load_with_selector` handles `Cpu` before it gets here (it
            // never creates an adapter); this is the guard for direct callers.
            bail!(
                "DeviceSelector::Cpu is the host backend, not a wgpu device — \
                 use WgpuAsr::load_on(.., DeviceSelector::Cpu) or `transcribe --cpu-dec`"
            );
        }
        let instance = wgpu::Instance::default();
        let adapters = instance.enumerate_adapters(selector.backends()).await;
        if adapters.is_empty() {
            bail!(
                "no wgpu adapters found (selector {selector:?}); try listing them first"
            );
        }

        // `Runtime { api, index }` picks the index-th *device* of that runtime,
        // ordered best-first (discrete before integrated), so `vulkan:1` is a
        // stable name for "the second GPU Vulkan can see" — the llama.cpp
        // convention.  `Auto` is that same ordering across all runtimes.
        let adapter = match &selector {
            DeviceSelector::Index(i) => adapters.get(*i).ok_or_else(|| {
                anyhow::anyhow!(
                    "device #{i} does not exist ({} adapter(s) visible: {})",
                    adapters.len(),
                    list_names(&adapters)
                )
            })?,
            sel => {
                let mut hits: Vec<&wgpu::Adapter> = adapters
                    .iter()
                    .filter(|a| sel.matches(&a.get_info()))
                    .collect();
                if hits.is_empty() {
                    let hint = if matches!(sel, DeviceSelector::Auto) {
                        String::new()
                    } else {
                        format!(" matching {sel:?}")
                    };
                    bail!("no adapter{hint} (visible: {})", list_names(&adapters));
                }
                hits.sort_by_key(|a| rank(&a.get_info()));
                match sel {
                    DeviceSelector::Runtime { index, .. } => hits.get(*index).copied().ok_or_else(
                        || {
                            anyhow::anyhow!(
                                "that runtime has {} device(s), index {index} is out of range",
                                hits.len()
                            )
                        },
                    )?,
                    _ => hits[0],
                }
            }
        };

        let info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("qwen3-asr-wgpu"),
                // Timestamp queries gate the per-op step profiler; SUBGROUP lets
                // `gemv` use warp shuffles instead of shared-memory butterflies.
                // All three are intersected with what the adapter reports, so an
                // adapter without them still gets a device (callers branch on
                // `Gpu::features`).  See `docs/wgpu-best-practices-audit.md`.
                required_features: features
                    & (wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                        | wgpu::Features::SUBGROUP
                        | wgpu::Features::PIPELINE_CACHE),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .context("request_device")?;

        // surface async validation errors / device loss instead of dying later
        // at an unrelated map with a bare "async map a buffer"
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            eprintln!("[wgpu uncaptured error] {e}");
        }));

        // Persisted pipeline cache: on D3D12 the compile is ~5.5 minutes for these
        // kernels (measured on Intel *and* NVIDIA; Vulkan does it in ~6 s), and
        // that is paid *per run* without a cache.  The key includes the adapter
        // and driver so a driver update gets a fresh cache, and `fallback: true`
        // means a stale or foreign cache is ignored rather than fatal.
        // Only when the adapter actually offers it — `create_pipeline_cache`
        // validates the feature and poisons the device otherwise.
        let cache_supported = features.contains(wgpu::Features::PIPELINE_CACHE);
        let (pipeline_cache, pipeline_cache_path) = match pipeline_cache_path(&info) {
            Some(path) if cache_supported => {
                let seed = std::fs::read(&path).ok();
                // SAFETY: the blob is driver-produced opaque data.  It is read
                // from our own cache directory, keyed on the adapter and driver
                // version, and `fallback: true` makes an unreadable or foreign
                // blob a cache miss instead of a failure.
                let cache = unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("pipeline_cache"),
                        data: seed.as_deref(),
                        fallback: true,
                    })
                };
                (Some(cache), Some(path))
            }
            _ => (None, None),
        };

        Ok(Self {
            device,
            queue,
            info,
            limits,
            features,
            pipeline_cache,
            pipeline_cache_path,
        })
    }

    /// Write the compiled pipelines back to disk so the next process starts
    /// warm.  Called once, after the decoder and the audio tower have built
    /// everything they are going to build.
    pub fn save_pipeline_cache(&self) -> Result<()> {
        let (Some(cache), Some(path)) = (&self.pipeline_cache, &self.pipeline_cache_path) else {
            return Ok(());
        };
        let Some(data) = cache.get_data() else {
            return Ok(());
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, &data)?;
        eprintln!(
            "[pipeline cache] wrote {} KiB to {}",
            data.len() / 1024,
            path.display()
        );
        Ok(())
    }

    /// One-line description for logs and reports.
    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}) | {} {} | wg workgroup storage {} B, maxStorageBufferBindingSize {} MiB",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            self.info.driver_info,
            self.limits.max_compute_workgroup_storage_size,
            self.limits.max_storage_buffer_binding_size / (1024 * 1024),
        )
    }

    /// Storage buffer sized `bytes` rounded up to a 16-byte multiple.
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 15) & !15;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn uniform(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 15) & !15;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Runtime-sized deferred write (queue ordering makes it visible to work
    /// submitted afterwards).  For multi-megabyte *load-time* transfers use
    /// [`Gpu::uploader`] instead — deferred staging must be bounded.
    pub fn upload(&self, buf: &wgpu::Buffer, data: &[u8]) {
        self.queue.write_buffer(buf, 0, data);
    }

    /// [`Gpu::upload`] at an explicit offset — for uniform buffers carrying one
    /// cfg per dispatch slot (see `decoder::prefill`'s slab cfgs).
    pub fn write_at(&self, buf: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buf, offset, data);
    }

    /// Submit an empty command buffer and wait for the queue to drain.  This
    /// retires every deferred `write_buffer` copy and frees their staging.
    pub fn flush(&self) -> Result<()> {
        let enc = self.device.create_command_encoder(&Default::default());
        self.queue.submit([enc.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll for flush")?;
        Ok(())
    }

    /// Start a bulk upload session (load-time weight transfer).
    pub fn uploader(&self) -> BulkUpload<'_> {
        BulkUpload { gpu: self, pending: 0 }
    }

    pub fn readback(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let size = (bytes + 3) & !3;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size.max(4));
        self.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll for readback")?;
        rx.recv().context("map callback dropped")?.context("map buffer")?;
        let mut data = slice.get_mapped_range()?.to_vec();
        drop(slice);
        staging.unmap();
        data.truncate(bytes as usize);
        Ok(data)
    }

    /// Compile a WGSL module + compute pipeline, surfacing validation errors
    /// instead of letting them turn into an opaque panic later.  `layout`
    /// attaches an explicit pipeline layout — required whenever one bind group
    /// is shared across sibling pipelines, because wgpu's implicit layouts are
    /// pipeline-exclusive.
    pub fn pipeline(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<wgpu::ComputePipeline> {
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: self.pipeline_cache.as_ref(),
        });
        let err = pollster::block_on(guard.pop());
        if let Some(e) = err {
            bail!("pipeline {label} failed validation: {e}");
        }
        Ok(pipe)
    }
}

/// Where the pipeline cache for this adapter lives: one file per
/// (vendor, device, backend, driver) under the user's cache directory.
///
/// Per-adapter rather than global because the blob is only valid for the driver
/// that produced it, and keyed on the driver version so a driver update does not
/// silently reuse a stale one.
fn pipeline_cache_path(info: &wgpu::AdapterInfo) -> Option<std::path::PathBuf> {
    let root = std::env::var_os("QASR_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from))
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .or_else(|| Some(std::env::temp_dir()))?;
    let key = format!(
        "{:04x}-{:04x}-{:?}-{}",
        info.vendor,
        info.device,
        info.backend,
        info.driver_info.replace(['\\', '/', ':', ' '], "_")
    );
    Some(root.join("qwen3-asr-wgpu").join(format!("{key}.pipeline_cache")))
}

/// Outstanding deferred-copy bytes tolerated before a flush.
///
/// wgpu defers every `queue.write_buffer` to the next submit and keeps the
/// device-side staging alive until then.  Letting a multi-GiB model load
/// accumulate into one submit overflows VRAM on WDDM drivers: copies silently
/// land as zeros and the device is lost on the next dispatch (observed on
/// Pascal for a 3.4 GiB load; 1.2 GiB happened to survive).  The CUDA engine
/// never holds staging because `htod` copies are synchronous — this budget is
/// the wgpu counterpart of that discipline: stage a bounded amount, then
/// submit + wait, which retires the staging, before staging more.
const STAGING_BUDGET: u64 = 256 << 20;

/// Load-time upload session with bounded outstanding staging.
///
/// Route every model-load transfer through one of these (`Gpu::uploader`);
/// runtime-sized writes (rope tables, token slots, KV seeding) stay on the
/// deferred [`Gpu::upload`] path, where queue ordering already guarantees
/// visibility and the volumes are tiny.
pub struct BulkUpload<'a> {
    gpu: &'a Gpu,
    pending: u64,
}

impl<'a> BulkUpload<'a> {
    /// Allocate a 16B-padded storage buffer (as [`Gpu::storage`]).
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.gpu.storage(label, bytes)
    }

    /// Allocate a uniform buffer (as [`Gpu::uniform`]).
    pub fn uniform(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.gpu.uniform(label, bytes)
    }

    /// Stage `data` into `buf`.  Pieces larger than the budget are split so a
    /// single staging buffer can never exceed it.
    pub fn upload(&mut self, buf: &wgpu::Buffer, data: &[u8]) -> Result<()> {
        let budget = (STAGING_BUDGET as usize).max(1);
        let mut off = 0usize;
        for piece in data.chunks(budget) {
            if self.pending + piece.len() as u64 > STAGING_BUDGET {
                self.gpu.flush()?;
                self.pending = 0;
            }
            self.gpu.queue.write_buffer(buf, off as u64, piece);
            self.pending += piece.len() as u64;
            off += piece.len();
        }
        Ok(())
    }

    /// Retire whatever is still staged.  Call before the first real dispatch.
    pub fn finish(self) -> Result<()> {
        if self.pending > 0 {
            self.gpu.flush()?;
        }
        Ok(())
    }
}
