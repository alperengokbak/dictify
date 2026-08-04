import threading
import wave

import numpy as np
import sounddevice as sd


def peak_dbfs(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0:
        return -120.0
    return 20 * np.log10(peak)


def estimate_noise_floor_dbfs(
    samples: np.ndarray, sample_rate: int, floor_fraction: float = 0.2
) -> float:
    frame_len = max(1, int(sample_rate * 0.02))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return peak_dbfs(samples)
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    frame_rms.sort()
    n_floor = max(1, int(n_frames * floor_fraction))
    mean_rms = float(np.mean(frame_rms[:n_floor]))
    if mean_rms <= 0:
        return -120.0
    return 20 * np.log10(mean_rms)


def is_silent(
    samples: np.ndarray,
    sample_rate: int,
    peak_floor_dbfs: float = -55.0,
    rise_db: float = 10.0,
) -> bool:
    peak = peak_dbfs(samples)
    if peak < peak_floor_dbfs:
        return True
    noise_floor = estimate_noise_floor_dbfs(samples, sample_rate)
    return bool((peak - noise_floor) < rise_db)


def record(stop_event: threading.Event, sample_rate: int = 16000) -> np.ndarray:
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32", callback=callback
    ):
        stop_event.wait()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0).flatten()


def save_wav(samples: np.ndarray, sample_rate: int, path: str) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
