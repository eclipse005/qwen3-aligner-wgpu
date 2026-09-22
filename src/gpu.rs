//! Device / queue wrapper and buffer plumbing.
//!
//! Two conventions, enforced by the constructors below:
//!
//! * storage buffers are always allocated 16-byte padded, so a `array<vec4<u32>>`
//!   view over an f16 payload never runs off the end;
//! * activation tensors are f16, indexed as two halves per `u32`.

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
/// The choice is explicit and enumerable rather than "whatever wgpu hands back":
/// a caller can run on the integrated GPU while the discrete one is busy, pin a
/// backend, or address the same machine's adapters by index.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    /// The default: the best GPU, on the runtime this engine is tuned for
    /// (Vulkan first, then Metal, then D3D12, then GL — see [`rank`]).
    #[default]
    Auto,
    /// A compute runtime and which device of it: `Runtime { api: Vulkan, index: 1 }`
    /// for `vulkan:1`.  This is the axis users mean by "backend".
    ///
    /// The vendor is *not* part of the choice: an NVIDIA card can be driven
    /// through Vulkan or D3D12, and those are different code paths.
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
/// `cpu` is our own implementation (see [`DeviceSelector::Cpu`]).  On Windows
/// wgpu drives the GPU through D3D12 **compute**, not DirectML.
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

    fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
        match self {
            Self::Auto => true,
            Self::Name(n) => info.name.to_lowercase().contains(n),
            Self::Index(_) | Self::Cpu => true,
            Self::Runtime { api, .. } => info.backend == *api,
        }
    }
}

/// The runtimes in the order [`rank`] prefers them.
const API_RANK_ORDER: [wgpu::Backend; 5] = [
    wgpu::Backend::Vulkan,
    wgpu::Backend::Metal,
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
    wgpu::Backend::BrowserWebGpu,
];

/// An instance that only knows about `backends`.
///
/// `Instance::default()` enables every runtime this build has, and creating it
/// is where the loaders of those runtimes get loaded: **127 ms** on this machine
/// against **20 ms** for a Vulkan-only instance (Vulkan's own enumeration is
/// 2.6 ms, D3D12's is 924 ms — both after the instance exists).  Flags and
/// backend options stay the defaults `Instance::default()` uses.
fn instance_for(backends: wgpu::Backends) -> wgpu::Instance {
    if backends == wgpu::Backends::all() {
        return wgpu::Instance::default();
    }
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    wgpu::Instance::new(desc)
}

/// Adapters `selector` needs to see, gathered no more widely than that.
///
/// This is startup cost, and on Windows it is not small: a default instance plus
/// a D3D12 enumeration is **~1.05 s**, against 20 ms + 2.6 ms for a Vulkan-only
/// instance — all of it before a single weight is read.  So:
///
/// * a named runtime gets an instance of itself and enumerates only itself;
/// * a raw index or a name substring is defined over the whole list, so those
///   still take the full instance and the full enumeration;
/// * `Auto` walks the runtimes in [`rank`] order — instance and all — and stops
///   as soon as a discrete GPU turns up: a later runtime could then only tie on
///   device class, and ties go to the earlier runtime, so the pick is the one
///   the full enumeration would have made.  A machine whose Vulkan has only an
///   iGPU (or nothing) still walks on, because there D3D12 *can* change the
///   answer.
async fn adapters_for(selector: &DeviceSelector) -> Vec<wgpu::Adapter> {
    match selector {
        DeviceSelector::Runtime { api, .. } => {
            let b = wgpu::Backends::from(*api);
            instance_for(b).enumerate_adapters(b).await
        }
        DeviceSelector::Index(_) | DeviceSelector::Name(_) => instance_for(wgpu::Backends::all())
            .enumerate_adapters(wgpu::Backends::all())
            .await,
        DeviceSelector::Auto => {
            let mut all = Vec::new();
            for api in API_RANK_ORDER {
                let b = wgpu::Backends::from(api);
                let mut found = instance_for(b).enumerate_adapters(b).await;
                let discrete = found
                    .iter()
                    .any(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
                all.append(&mut found);
                if discrete {
                    break;
                }
            }
            all
        }
        DeviceSelector::Cpu => Vec::new(),
    }
}

/// Default-selection order: discrete before integrated before virtual/CPU, then
/// by graphics API (Vulkan, Metal, D3D12, GL).
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

/// One enumerated adapter: what you need to choose a device, without creating
/// one.
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
    /// `ID3D12PipelineLibrary`).
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
/// [`DeviceSelector::Index`] indexes into.  One entry per *(device, graphics
/// API)* pair: the same physical GPU appears once per API it is reachable
/// through.
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
/// `vulkan:0`, `dx12:1`, …  The runtime is the axis; the vendor and device
/// class are information.
#[derive(Debug, Clone)]
pub struct DeviceTarget {
    /// `<runtime>:<index>`, e.g. `vulkan:1` — feed it back via `--device`.
    pub spec: String,
    pub info: DeviceInfo,
    /// True when [`DeviceSelector::Auto`] would pick this target.
    pub is_default: bool,
}

impl DeviceTarget {
    /// `vulkan:0  NVIDIA … (NVIDIA, dGPU, driver 572.75) binding 2047 MiB, subgroup 32`
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
        let t = std::time::Instant::now();
        let adapters = adapters_for(&selector).await;
        crate::load_trace::note("gpu: instance + enumerate", t);
        if adapters.is_empty() {
            bail!(
                "no wgpu adapters found (selector {selector:?}); try listing them first"
            );
        }

        // `Runtime { api, index }` picks the index-th *device* of that runtime,
        // ordered best-first (discrete before integrated), so `vulkan:1` is a
        // stable name for "the second GPU Vulkan can see".  `Auto` uses that
        // same ordering across all runtimes.
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

        let t = std::time::Instant::now();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("qwen3-aligner-wgpu"),
                // Timestamp queries gate the per-op step profiler; SUBGROUP lets
                // `gemv` use warp shuffles instead of shared-memory butterflies.
                // All three are intersected with what the adapter reports, so an
                // adapter without them still gets a device.
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
        crate::load_trace::note("gpu: request_device", t);

        // surface async validation errors / device loss instead of dying later
        // at an unrelated map with a bare "async map a buffer"
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            eprintln!("[wgpu uncaptured error] {e}");
        }));

        // Persisted pipeline cache: the key includes the adapter and driver so a
        // driver update gets a fresh cache, and `fallback: true` makes a stale or
        // foreign cache a miss rather than fatal.  Only created when the adapter
        // offers the feature — `create_pipeline_cache` validates it and poisons
        // the device otherwise.
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
        self.pump()?;
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll for flush")?;
        Ok(())
    }

    /// Submit without waiting.
    ///
    /// `queue.write_buffer` only *stages* its copies; they do not start moving
    /// until the next submission.  A load that stages a whole model and submits
    /// once therefore leaves the bus idle through every conversion and every
    /// memcpy and then pays the entire transfer at the end — the pump is what
    /// lets the DMA of one chunk run against the host preparing the next.
    pub fn pump(&self) -> Result<()> {
        let enc = self.device.create_command_encoder(&Default::default());
        self.queue.submit([enc.finish()]);
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
        // `BufferSlice` is `Copy`, so an explicit `drop` here would be a no-op —
        // the map is released by `staging.unmap()` below, not by dropping this.
        staging.unmap();
        data.truncate(bytes as usize);
        Ok(data)
    }

    /// Compile a WGSL module + compute pipeline, surfacing validation errors
    /// instead of letting them turn into an opaque panic.  `layout` attaches an
    /// explicit pipeline layout — required when one bind group is shared across
    /// sibling pipelines, since wgpu's implicit layouts are pipeline-exclusive.
    ///
    /// The run's 16-bit storage format is prepended here, once, in front of
    /// whatever the caller wrote: every kernel unpacks and packs through
    /// `unpack_h` / `pack_h`, so the format is a single choice for the whole
    /// process ([`crate::shaders::half`]) rather than a parameter on ~33 shader
    /// builders.
    pub fn pipeline(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<wgpu::ComputePipeline> {
        let source = format!("{}{wgsl}", crate::shaders::half().prelude());
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
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
/// (vendor, device, backend, driver) under the user's cache directory.  The blob
/// is only valid for the driver that produced it, so the driver version is part
/// of the key.
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
    Some(root.join("qwen3-aligner-wgpu").join(format!("{key}.pipeline_cache")))
}

/// Outstanding deferred-copy bytes tolerated before a flush.  wgpu defers every
/// `queue.write_buffer` to the next submit and holds the staging until then, so a
/// multi-GiB load in one submit overflows VRAM on WDDM; stage a bounded amount,
/// then submit and wait to retire it.
const STAGING_BUDGET: u64 = 256 << 20;

/// Load-time upload session with bounded outstanding staging.  Route every
/// model-load transfer through one of these (`Gpu::uploader`); runtime-sized
/// writes (rope tables, token slots, KV seeding) use the deferred [`Gpu::upload`]
/// path, where queue ordering already guarantees visibility.
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
    ///
    /// Every [`STAGING_BUDGET`] bytes the staged copies are *submitted* — not
    /// waited on — so their DMA runs while the caller converts and stages the
    /// next chunk.  Without that the whole transfer lands on the final
    /// [`Self::finish`]: see [`Gpu::pump`].  The budget is what bounds how much
    /// can be in flight.
    pub fn upload(&mut self, buf: &wgpu::Buffer, data: &[u8]) -> Result<()> {
        self.upload_at(buf, 0, data)
    }

    /// [`Self::upload`] at an explicit offset — so a fused matrix (`q|k|v`,
    /// `gate|up`) can be written part by part instead of being concatenated on
    /// the host first.  The offset must be a multiple of 4 (wgpu's copy
    /// alignment); every f16 row is a multiple of 2, and every call site here
    /// passes whole matrices.
    pub fn upload_at(&mut self, buf: &wgpu::Buffer, offset: u64, data: &[u8]) -> Result<()> {
        anyhow::ensure!(offset % 4 == 0, "upload offset {offset} is not 4-byte aligned");
        let budget = (STAGING_BUDGET as usize).max(1);
        let mut off = offset;
        for piece in data.chunks(budget) {
            if self.pending + piece.len() as u64 > STAGING_BUDGET {
                self.pump()?;
            }
            let t = std::time::Instant::now();
            self.gpu.queue.write_buffer(buf, off, piece);
            crate::load_trace::transfer::add_write(
                t.elapsed().as_nanos() as u64,
                piece.len() as u64,
            );
            self.pending += piece.len() as u64;
            off += piece.len() as u64;
        }
        Ok(())
    }

    /// Allocate a storage buffer and fill it from `pieces`, each `(offset, bytes)`.
    ///
    /// The pieces come straight out of [`crate::weights`] in the layout the
    /// kernels want, so a fused matrix is never materialised twice: an f16
    /// checkpoint contributes its own mapped bytes, a bf16 one a single
    /// narrowing pass into the destination layout.
    pub fn upload_pieces(
        &mut self,
        label: &str,
        pieces: &[(u64, bytes::Bytes)],
    ) -> Result<wgpu::Buffer> {
        let total: u64 = pieces.iter().map(|(_, b)| b.len() as u64).sum();
        let buf = self.storage(label, total);
        for (off, data) in pieces {
            self.upload_at(&buf, *off, data)?;
        }
        Ok(buf)
    }

    /// Retire whatever is still staged.  Call before the first real dispatch.
    pub fn finish(mut self) -> Result<()> {
        if self.pending > 0 {
            self.flush()?;
        }
        Ok(())
    }

    /// Submit the staged copies without waiting for them.
    fn pump(&mut self) -> Result<()> {
        let t = std::time::Instant::now();
        self.gpu.pump()?;
        crate::load_trace::transfer::add_submit(t.elapsed().as_nanos() as u64);
        self.pending = 0;
        Ok(())
    }

    /// Submit and wait — the last one, where the transfer really is the wait.
    fn flush(&mut self) -> Result<()> {
        let t = std::time::Instant::now();
        self.gpu.flush()?;
        crate::load_trace::transfer::add_wait(t.elapsed().as_nanos() as u64);
        self.pending = 0;
        Ok(())
    }
}
