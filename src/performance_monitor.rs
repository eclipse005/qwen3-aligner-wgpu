//! 性能监控和分析工具
//! 
//! 提供详细的性能指标收集和报告功能

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// 性能计数器
#[derive(Debug, Clone)]
pub struct PerfCounter {
    pub count: u64,
    pub total_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub last_duration: Duration,
}

impl PerfCounter {
    fn new() -> Self {
        Self {
            count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            last_duration: Duration::ZERO,
        }
    }

    fn record(&mut self, duration: Duration) {
        self.count += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.last_duration = duration;
    }

    pub fn avg_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_duration.as_secs_f64() * 1000.0 / self.count as f64
        }
    }

    pub fn total_ms(&self) -> f64 {
        self.total_duration.as_secs_f64() * 1000.0
    }

    pub fn min_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.min_duration.as_secs_f64() * 1000.0
        }
    }

    pub fn max_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max_duration.as_secs_f64() * 1000.0
        }
    }

    pub fn last_ms(&self) -> f64 {
        self.last_duration.as_secs_f64() * 1000.0
    }
}

/// 全局性能监控器
pub struct PerformanceMonitor {
    counters: Arc<Mutex<HashMap<String, PerfCounter>>>,
    enabled: bool,
}

impl PerformanceMonitor {
    pub fn new(enabled: bool) -> Self {
        Self {
            counters: Arc::new(Mutex::new(HashMap::new())),
            enabled,
        }
    }

    /// 创建一个定时器
    pub fn timer(&self, name: &str) -> PerfTimer {
        PerfTimer {
            name: name.to_string(),
            start: Instant::now(),
            monitor: self.clone(),
        }
    }

    /// 记录一次测量
    fn record(&self, name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }
        
        let mut counters = self.counters.lock().unwrap();
        let counter = counters.entry(name.to_string()).or_insert_with(PerfCounter::new);
        counter.record(duration);
    }

    /// 生成性能报告
    pub fn report(&self) -> String {
        if !self.enabled {
            return "Performance monitoring is disabled".to_string();
        }
        
        let counters = self.counters.lock().unwrap();
        let mut report = String::new();
        
        report.push_str("┌─────────────────────────────────────────────────────────────────┐\n");
        report.push_str("│               Performance Report                                │\n");
        report.push_str("├─────────────────────────────────────────────────────────────────┤\n");
        report.push_str("│ Metric              │ Count │  Avg   │  Min   │  Max   │ Total │\n");
        report.push_str("├─────────────────────────────────────────────────────────────────┤\n");
        
        // 按名称排序
        let mut items: Vec<_> = counters.iter().collect();
        items.sort_by_key(|(name, _)| name.as_str());
        
        for (name, counter) in items {
            report.push_str(&format!(
                "│ {:<19} │ {:>5} │ {:>6.1} │ {:>6.1} │ {:>6.1} │ {:>5.0} │\n",
                truncate_string(name, 19),
                counter.count,
                counter.avg_ms(),
                counter.min_ms(),
                counter.max_ms(),
                counter.total_ms()
            ));
        }
        
        report.push_str("└─────────────────────────────────────────────────────────────────┘\n");
        report.push_str("  (Times in milliseconds)\n");
        
        report
    }

    /// 重置所有计数器
    pub fn reset(&self) {
        let mut counters = self.counters.lock().unwrap();
        counters.clear();
    }

    /// 获取特定指标
    pub fn get_counter(&self, name: &str) -> Option<PerfCounter> {
        let counters = self.counters.lock().unwrap();
        counters.get(name).cloned()
    }
}

impl Clone for PerformanceMonitor {
    fn clone(&self) -> Self {
        Self {
            counters: Arc::clone(&self.counters),
            enabled: self.enabled,
        }
    }
}

/// RAII 风格的性能计时器
pub struct PerfTimer {
    name: String,
    start: Instant,
    monitor: PerformanceMonitor,
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.monitor.record(&self.name, duration);
    }
}

/// 截断字符串以适应固定宽度
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// 全局性能监控器实例
lazy_static::lazy_static! {
    pub static ref GLOBAL_PERF_MONITOR: PerformanceMonitor = {
        let enabled = std::env::var("QWEN3_PERF_MONITOR")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        PerformanceMonitor::new(enabled)
    };
}

/// 便捷宏：创建一个作用域内的性能计时器
#[macro_export]
macro_rules! perf_timer {
    ($name:expr) => {
        let _timer = $crate::performance_monitor::GLOBAL_PERF_MONITOR.timer($name);
    };
}

/// 系统信息收集
#[cfg(target_os = "macos")]
pub fn collect_system_info() -> String {
    use std::process::Command;
    
    let mut info = String::new();
    
    // CPU 信息
    if let Ok(output) = Command::new("sysctl").args(&["-n", "machdep.cpu.brand_string"]).output() {
        if let Ok(cpu) = String::from_utf8(output.stdout) {
            info.push_str(&format!("CPU: {}\n", cpu.trim()));
        }
    }
    
    // CPU 核心数
    if let Ok(output) = Command::new("sysctl").args(&["-n", "hw.ncpu"]).output() {
        if let Ok(cores) = String::from_utf8(output.stdout) {
            info.push_str(&format!("Cores: {}\n", cores.trim()));
        }
    }
    
    // 内存
    if let Ok(output) = Command::new("sysctl").args(&["-n", "hw.memsize"]).output() {
        if let Ok(mem) = String::from_utf8(output.stdout) {
            if let Ok(bytes) = mem.trim().parse::<u64>() {
                info.push_str(&format!("Memory: {:.1} GB\n", bytes as f64 / 1_073_741_824.0));
            }
        }
    }
    
    // GPU 信息（Metal）
    info.push_str("GPU: Apple Silicon (Metal)\n");
    
    info
}

#[cfg(not(target_os = "macos"))]
pub fn collect_system_info() -> String {
    "System info collection not implemented for this platform\n".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_perf_counter() {
        let mut counter = PerfCounter::new();
        assert_eq!(counter.count, 0);
        
        counter.record(Duration::from_millis(100));
        assert_eq!(counter.count, 1);
        assert!((counter.avg_ms() - 100.0).abs() < 1.0);
        
        counter.record(Duration::from_millis(200));
        assert_eq!(counter.count, 2);
        assert!((counter.avg_ms() - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new(true);
        
        {
            let _timer = monitor.timer("test_operation");
            thread::sleep(Duration::from_millis(10));
        }
        
        let counter = monitor.get_counter("test_operation").unwrap();
        assert_eq!(counter.count, 1);
        assert!(counter.avg_ms() >= 10.0);
        
        let report = monitor.report();
        assert!(report.contains("test_operation"));
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("short", 10), "short");
        assert_eq!(truncate_string("very_long_string", 10), "very_lo...");
    }
}
