"""Compare wgpu probe dump vs torch reference, layer by layer."""
import sys, os
import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else "q06_15s_en"
ref_dir = os.path.join("wgpu", "golden", tag + "_ref")
got_dir = os.path.join("wgpu", "golden", tag + "_wgpu_probe")

ops = ["norm1", "qkv", "q_out", "attn_out", "norm2", "gate_up", "activated", "h"]
print(f"{'layer':<6}" + "".join(f"{op:>12}" for op in ops))
for l in range(28):
    row = f"{l:<6}"
    for op in ops:
        a = np.load(os.path.join(ref_dir, f"L{l:02d}_{op}.npy"))
        b = np.fromfile(os.path.join(got_dir, f"L{l:02d}_{op}.bin"), dtype="<f2").astype(np.float32)
        d = np.abs(a - b).max() if a.shape == b.shape else float("nan")
        rel = d / max(np.abs(a).max(), 1e-9)
        row += f"{d:>7.3f}({rel:.0e})"
    print(row)

a = np.load(os.path.join(ref_dir, "final_norm.npy"))
b = np.fromfile(os.path.join(got_dir, "final_norm.bin"), dtype="<f2").astype(np.float32)
print("final_norm max|d| =", np.abs(a - b).max())
print("token:", open(os.path.join(got_dir, "token.txt")).read())
