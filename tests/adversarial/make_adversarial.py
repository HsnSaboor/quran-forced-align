"""Build adversarial repeat clips from tests/fixtures/test_B_repeat_pause.wav.

Carrier: first utterance of the 4-word phrase WORDS[3:7] (oracle 3.720s-7.800s
of test_B), i.e. a NATURAL single recitation of the phrase -- the reference
fed to the harness is exactly WORDS_0_7-minus? No: harness reference must be
7 words wide. Simpler design: carrier = WHOLE test_B clip trimmed to the
first utterance + the *second-utterance phrase* region re-synthesized with
edits. In practice the cleanest recipe: carrier = full test_B first
utterance (0-8.0s) + edited copy of the second-utterance phrase
(8.480-12.560s => words[3:7] second occurrence) appended with a gap.

Oracle second-copy bounds (test_B, 16kHz): start=8.480s end=12.560s.
Cut a hair inside VAD edges to avoid blank frames: SEG = [8.48, 12.56].
Actually safer: [8.40, 12.61] (clip end) then all splices share identical phoneme
content -- edits (speed/pitch/gain/noise/reverse) applied to SEG only.

Per-case mix recipes (ffmpeg, 16kHz mono s16 wav). BASE = first-utterance
carrier: test_B[0, 8.0s] (covers words0-6 + ~0.2s of the natural pause;
fades at the joint).

Speed without pitch preservation ("resample method"): atempo changes both
tempo AND pitch when followed by aresample back? No -- atempo preserves
pitch by design. True "no pitch preservation" = asetrate + aresample:
  asetrate=16000*S,aresample=16000  (changes tempo AND pitch, classic
  resample-speedup). Then acompensate length: S>1 shortens.
Pitch-preserved = atempo=S (WSOLA).

Gaps: 0.5s digital silence unless the case says otherwise.
For speed-S copies the copy is shorter; gap still 0.5s.

All outputs: 16kHz mono s16.
"""
import os, shutil, subprocess, sys

FIX = "tests/fixtures/test_B_repeat_pause.wav"
OUT = "tests/adversarial"
SR = 16000

if shutil.which("ffmpeg") is None:
    sys.exit("ffmpeg not found -- install ffmpeg 6.x to rebuild adversarial clips (checked-in wavs under tests/adversarial/ remain usable without it)")

# oracle bounds (sec)
COPY_S, COPY_E = 8.40, 12.61   # second-utterance phrase incl. small pads
CARRIER_E = 8.0                # carrier = [0, 8.0]

def run(cmd):
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:]); sys.exit(1)

def base_extract(tmp):
    # carrier + copy segment as wavs
    run(["ffmpeg","-y","-v","error","-ss","0","-to",str(CARRIER_E),"-i",FIX,
         "-ar",str(SR),"-ac","1","-c:a","pcm_s16le",f"{tmp}/carrier.wav"])
    run(["ffmpeg","-y","-v","error","-ss",str(COPY_S),"-to",str(COPY_E),"-i",FIX,
         "-ar",str(SR),"-ac","1","-c:a","pcm_s16le",f"{tmp}/copy.wav"])

def speed_filter(speed, pitch_preserved):
    if pitch_preserved:
        return f"atempo={speed}"
    else:
        # resample speedup: pitch rises with speed
        return f"asetrate={SR}*{speed},aresample={SR}"

def mix_concat(tmp, copy_wav, out, gap=0.5, crossfade=None):
    """concat carrier + gap + copy (or crossfade carrier/copy with no gap)."""
    if crossfade is not None:
        run(["ffmpeg","-y","-v","error","-i",f"{tmp}/carrier.wav","-i",copy_wav,
             "-filter_complex",
             f"[0:a][1:a]acrossfade=d={crossfade}[aout]",
             "-map","[aout]","-ar",str(SR),"-ac","1","-c:a","pcm_s16le",out])
    else:
        run(["ffmpeg","-y","-v","error","-i",f"{tmp}/carrier.wav","-i",copy_wav,
             "-filter_complex",
             f"aevalsrc=0:d={gap}:s={SR}[sil];[0:a][sil][1:a]concat=n=3:v=0:a=1[aout]",
             "-map","[aout]","-ar",str(SR),"-ac","1","-c:a","pcm_s16le",out])

def make_case(tmp, name, filt=None, gain_db=None, noise=None, gap=0.5, crossfade=None,
              reverse=False, speed_copy=None):
    copy = f"{tmp}/copy.wav"
    mod = copy
    chain = []
    if filt: chain.append(filt)
    if reverse: chain.append("areverse")
    if gain_db is not None:
        mod = f"{tmp}/{name}_mod.wav"
        f = ",".join(chain + [f"volume={gain_db}dB"]) if chain else f"volume={gain_db}dB"
        run(["ffmpeg","-y","-v","error","-i",copy,"-af",f,
             "-ar",str(SR),"-ac","1","-c:a","pcm_s16le",mod])
    elif chain:
        mod = f"{tmp}/{name}_mod.wav"
        run(["ffmpeg","-y","-v","error","-i",copy,"-af",",".join(chain),
             "-ar",str(SR),"-ac","1","-c:a","pcm_s16le",mod])
    if noise == "white":
        nz = f"{tmp}/{name}_nz.wav"
        # pink-ish low white noise mixed at low level
        run(["ffmpeg","-y","-v","error","-i",mod,
             "-filter_complex",
             f"anoisesrc=d=60:c=white:r={SR}:a=0.015,volume=0.5[nz];[0:a][nz]amix=inputs=2:duration=first:dropout_transition=0[m]",
             "-map","[m]","-ar",str(SR),"-ac","1","-c:a","pcm_s16le",nz])
        mod = nz
    elif noise == "hum":
        nz = f"{tmp}/{name}_nz.wav"
        run(["ffmpeg","-y","-v","error","-i",mod,
             "-filter_complex",
             f"sine=frequency=120:sample_rate={SR}:duration=60,volume=0.06[hum];[0:a][hum]amix=inputs=2:duration=first:dropout_transition=0[m]",
             "-map","[m]","-ar",str(SR),"-ac","1","-c:a","pcm_s16le",nz])
        mod = nz
    mix_concat(tmp, mod, f"{OUT}/{name}.wav", gap=gap, crossfade=crossfade)

def triple(tmp, name):
    c = f"{tmp}/copy.wav"
    run(["ffmpeg","-y","-v","error",
         "-i",f"{tmp}/carrier.wav","-i",c,"-i",c,
         "-filter_complex",
         "aevalsrc=0:d=0.5:s=16000[s1];aevalsrc=0:d=0.5:s=16000[s2];"
         "[0:a][s1][1:a][s2][2:a]concat=n=5:v=0:a=1[aout]",
         "-map","[aout]","-ar","16000","-ac","1","-c:a","pcm_s16le",f"{OUT}/{name}.wav"])

def main():
    tmp = "/tmp/qlab/scratch/advbuild"
    os.makedirs(tmp, exist_ok=True)
    base_extract(tmp)
    # (a) pitch-preserved speedups
    make_case(tmp, "adv_125_atempo", filt=speed_filter(1.25, True))
    make_case(tmp, "adv_150_atempo", filt=speed_filter(1.5, True))
    make_case(tmp, "adv_200_atempo", filt=speed_filter(2.0, True))
    # (b) resample speedups (pitch shift up)
    make_case(tmp, "adv_125_resample", filt=speed_filter(1.25, False))
    make_case(tmp, "adv_150_resample", filt=speed_filter(1.5, False))
    # (c) zero gap
    make_case(tmp, "adv_zerogap", gap=0.0)
    # (d) 50ms crossfade overlap
    make_case(tmp, "adv_xfade50", crossfade=0.05)
    # (e) triple
    triple(tmp, "adv_triple")
    # (f) tempo-mismatched inserted copy: carrier natural + copy at 1.3x atempo
    make_case(tmp, "adv_tempo130", filt=speed_filter(1.3, True))
    # (g) partial: first 2 of 4 words only. word bounds of 2nd utterance:
    # w3 8.48-9.12, w4 9.24-9.40 -> cut [8.40, 9.60]
    run(["ffmpeg","-y","-v","error","-ss","8.40","-to","9.60","-i",FIX,
         "-ar","16000","-ac","1","-c:a","pcm_s16le",f"{tmp}/partial.wav"])
    mix_concat(tmp, f"{tmp}/partial.wav", f"{OUT}/adv_partial2of4.wav", gap=0.5)
    # (h) paraphrase decoy: DIFFERENT words with similar cadence:
    # use carrier's own words[0:2] region (0.32-2.88, different words, 2.5s)
    # spliced after full carrier -> acoustically similar speaker, wrong words
    run(["ffmpeg","-y","-v","error","-ss","0.30","-to","2.90","-i",FIX,
         "-ar","16000","-ac","1","-c:a","pcm_s16le",f"{tmp}/decoy.wav"])
    mix_concat(tmp, f"{tmp}/decoy.wav", f"{OUT}/adv_paraphrase.wav", gap=0.5)
    # (i) ayah-boundary crossing: append FIRST-phrase words of a different
    # ayah context -- use test_A clip's tail? Simplest honest version:
    # copy = words[0:2] region (different phrase from same reciter) =
    # same as decoy but EXPECT characterization. Instead: repeat of words
    # [5:7] (last 2 words) + words[0:1] of "next ayah" simulated by words0-1.
    # Build: carrier + [w5..w6 second-utterance slice 9.60-12.61] + [w0w1 0.30-2.90]
    run(["ffmpeg","-y","-v","error","-ss","9.60","-to","12.61","-i",FIX,
         "-ar","16000","-ac","1","-c:a","pcm_s16le",f"{tmp}/tail2.wav"])
    run(["ffmpeg","-y","-v","error",
         "-i",f"{tmp}/carrier.wav","-i",f"{tmp}/tail2.wav","-i",f"{tmp}/decoy.wav",
         "-filter_complex",
         "aevalsrc=0:d=0.5:s=16000[s1];aevalsrc=0:d=0.5:s=16000[s2];"
         "[0:a][s1][1:a][s2][2:a]concat=n=5:v=0:a=1[aout]",
         "-map","[aout]","-ar","16000","-ac","1","-c:a","pcm_s16le",f"{OUT}/adv_ayahx.wav"])
    # (j) attenuated
    make_case(tmp, "adv_att6", gain_db=-6)
    make_case(tmp, "adv_att12", gain_db=-12)
    # (k) noise / hum
    make_case(tmp, "adv_noise", noise="white")
    make_case(tmp, "adv_hum", noise="hum")
    # (l) reversed
    make_case(tmp, "adv_reversed", reverse=True)
    print("BUILD DONE")
    import glob
    for f in sorted(glob.glob(f"{OUT}/*.wav")):
        print("  ", f)

if __name__ == "__main__":
    main()
