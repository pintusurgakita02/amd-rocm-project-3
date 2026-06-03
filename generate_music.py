import torch, torchaudio, argparse
def generate(prompt, dur=10, sz="medium"):
    from audiocraft.models import MusicGen
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    m = MusicGen.get_pretrained(f"facebook/musicgen-{sz}", device=device)
    m.set_generation_params(duration=dur)
    wav = m.generate([prompt])
    torchaudio.save("out.wav", wav[0].cpu(), m.sample_rate)
    print(f"Saved out.wav ({dur}s)")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", type=int, default=10)
    a = p.parse_args()
    generate(a.prompt, a.duration)
