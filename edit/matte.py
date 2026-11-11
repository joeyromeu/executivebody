"""Run Robust Video Matting (ONNX) over a video and write the alpha matte as a lossless gray video."""
import subprocess, sys, numpy as np, onnxruntime as ort
src, out = sys.argv[1], sys.argv[2]
ss = (sys.argv[3] or None) if len(sys.argv) > 3 else None
nframes = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
W, H = 720, 1280
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(sys.argv[5] if len(sys.argv) > 5 else 'rvm.onnx', so)
dec = ["ffmpeg", "-v", "error"] + (["-ss", ss] if ss else []) + ["-i", src] + (["-frames:v", str(nframes)] if nframes else []) + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "ffv1", out]
pd = subprocess.Popen(dec, stdout=subprocess.PIPE)
pe = subprocess.Popen(enc, stdin=subprocess.PIPE)
rec = [np.zeros([1, 1, 1, 1], np.float32)] * 4
dr = np.array([0.4], np.float32)
i = 0
while True:
    buf = pd.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    x = (fr.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    fgr, pha, *rec = sess.run(None, {"src": x, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3], "downsample_ratio": dr})
    a = (np.clip(pha[0, 0], 0, 1) * 255).astype(np.uint8)
    pe.stdin.write(a.tobytes())
    i += 1
    if i % 150 == 0: print("frame", i, flush=True)
pe.stdin.close(); pe.wait(); print("done", i)
