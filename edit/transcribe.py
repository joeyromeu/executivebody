import json, sys
from faster_whisper import WhisperModel
m = WhisperModel("small.en", device="cpu", compute_type="int8")
import wave, numpy as np
wf=wave.open(sys.argv[1]); audio=np.frombuffer(wf.readframes(wf.getnframes()),dtype=np.int16).astype(np.float32)/32768.0
segs, info = m.transcribe(audio, word_timestamps=True, beam_size=5, vad_filter=False)
out=[]
for s in segs:
    words=[{"w":w.word,"s":round(w.start,2),"e":round(w.end,2)} for w in s.words]
    out.append({"s":round(s.start,2),"e":round(s.end,2),"text":s.text,"words":words})
    print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.text}", flush=True)
json.dump(out, open(sys.argv[2],"w"), indent=1)
