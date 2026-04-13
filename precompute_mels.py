from pathlib import Path
from multiprocessing import Pool, cpu_count

import torch
import torchaudio
from tqdm import tqdm

WAV_DIR = Path("data/LJSpeech-1.1/wavs")
MEL_DIR = Path("data/LJSpeech-1.1/mels")


def make_mel_transform():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        n_mels=80,
        win_length=1024,
        hop_length=256,
        f_min=0,
        f_max=8000,
        power=1.0,
        normalized=False,
        norm="slaney",
        mel_scale="slaney",
    )


mel_transform = None


def init_worker():
    global mel_transform
    mel_transform = make_mel_transform()


def process_file(wav_path):
    wav, sr = torchaudio.load(wav_path)
    mel = torch.log(torch.clip(mel_transform(wav), min=1e-5))
    mel = mel.squeeze(0)  # (n_mels, time)
    out_path = MEL_DIR / (wav_path.stem + ".pt")
    torch.save(mel, out_path)


def main():
    MEL_DIR.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(WAV_DIR.glob("*.wav"))
    print(f"Found {len(wav_files)} wav files, using {cpu_count()} workers")

    with Pool(processes=cpu_count(), initializer=init_worker) as pool:
        list(
            tqdm(pool.imap(process_file, wav_files, chunksize=64), total=len(wav_files))
        )

    print(f"Done. Mels saved to {MEL_DIR}/")


if __name__ == "__main__":
    main()
