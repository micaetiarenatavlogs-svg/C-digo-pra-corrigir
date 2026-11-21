#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RVC Voice Training - HuggingFace Space
PARTE 1/4

Inclui:
- armazenamento persistente;
- estado de jobs em memória;
- persistência assíncrona;
- logs por job;
- conversão de áudio;
- pré-processamento;
- segmentação;
- extração de F0.

As partes seguintes adicionam:
- HuBERT;
- FAISS;
- arquitetura RVC;
- treinamento;
- retomada por checkpoint;
- Gradio;
- downloads.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
import threading
import time
import re
import traceback
import warnings
import tempfile
import math
import random
import queue
import logging
import gc
import gradio as gr
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import librosa
import soundfile as sf
from scipy.io import wavfile
from scipy import signal

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

def _get_writable_storage_root() -> Path:
    """
    Prefere armazenamento persistente da Space.

    Ordem:
    1. /persistent - armazenamento persistente do Hugging Face;
    2. /data       - usado por algumas configurações de Spaces;
    3. /mnt/data   - fallback;
    4. /tmp        - último recurso.

    Se somente /tmp estiver disponível, os jobs não sobreviverão a um
    reinício do container.
    """
    candidates = (
        "/persistent",
        "/data",
        "/mnt/data",
        "/tmp",
    )

    for candidate in candidates:
        try:
            root = Path(candidate)
            root.mkdir(parents=True, exist_ok=True)

            test_file = root / ".rvc_write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)

            return root
        except Exception:
            continue

    raise RuntimeError("Nenhum diretório gravável foi encontrado.")


STORAGE_ROOT = _get_writable_storage_root()
BASE_DIR = STORAGE_ROOT / "rvc_training"

JOBS_DIR = BASE_DIR / "jobs"
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
PREVIEWS_DIR = BASE_DIR / "previews"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TEMP_DIR = BASE_DIR / "temp"
DOWNLOADS_DIR = BASE_DIR / "downloads"
ASSETS_DIR = BASE_DIR / "assets"
WEIGHTS_DIR = BASE_DIR / "weights"

for directory in (
    JOBS_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    LOGS_DIR,
    PREVIEWS_DIR,
    CHECKPOINTS_DIR,
    TEMP_DIR,
    DOWNLOADS_DIR,
    ASSETS_DIR,
    WEIGHTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

JOBS_FILE = JOBS_DIR / "jobs.json"

MIN_AUDIO_SECONDS = 180
MAX_AUDIO_SECONDS = 40 * 60
MIN_EPOCHS = 50
MAX_EPOCHS = 1000

# Não permita 100 treinos simultâneos: isso normalmente derruba a Space.
# Para GPU, use 1. Para CPU, 1 ou 2.
DEFAULT_MAX_CONCURRENT = 1 if torch.cuda.is_available() else 2
MAX_CONCURRENT_TRAINING = int(
    os.getenv("MAX_CONCURRENT_TRAINING", DEFAULT_MAX_CONCURRENT)
)

SR_TARGET = 40000

STATUS_WAITING = "Esperando"
STATUS_PREPROCESSING = "Pre-processando"
STATUS_EXTRACTING_F0 = "Extraindo F0"
STATUS_EXTRACTING_FEATURES = "Extraindo Features"
STATUS_TRAINING_INDEX = "Treinando Indice"
STATUS_TRAINING = "Treinando"
STATUS_DONE = "Concluido"
STATUS_FAILED = "Falha"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    DEVICE_LABEL = f"GPU ({torch.cuda.get_device_name(0)})"
else:
    DEVICE_LABEL = "CPU"

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("rvc_train")

# ============================================================================
# JOBS
# ============================================================================

JOBS_CACHE: Dict[str, dict] = {}
_JOBS_CACHE_LOCK = threading.RLock()
_DISK_WRITER_STOP = threading.Event()


def _atomic_json_write(path: Path, data: dict) -> None:
    """Escreve JSON de forma atômica para evitar jobs.json corrompido."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


def _disk_writer_daemon() -> None:
    """
    Persiste o estado em background.

    O treinamento nunca deve esperar por uma operação de disco feita pela UI.
    """
    while not _DISK_WRITER_STOP.wait(3.0):
        try:
            with _JOBS_CACHE_LOCK:
                snapshot = {
                    job_id: dict(job)
                    for job_id, job in JOBS_CACHE.items()
                }

            if snapshot:
                _atomic_json_write(JOBS_FILE, snapshot)

        except Exception:
            logger.exception("Erro ao persistir jobs em disco.")


def _load_from_disk() -> dict:
    if not JOBS_FILE.exists():
        return {}

    try:
        with JOBS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except Exception:
        logger.exception("Não foi possível carregar jobs.json.")
        return {}


def _init_cache() -> None:
    loaded_jobs = _load_from_disk()

    with _JOBS_CACHE_LOCK:
        JOBS_CACHE.clear()

        for job_id, job in loaded_jobs.items():
            if isinstance(job, dict):
                JOBS_CACHE[str(job_id)] = job


def get_jobs() -> Dict[str, dict]:
    with _JOBS_CACHE_LOCK:
        return {
            job_id: dict(job)
            for job_id, job in JOBS_CACHE.items()
        }


def get_job(job_id: str) -> Optional[dict]:
    with _JOBS_CACHE_LOCK:
        job = JOBS_CACHE.get(job_id)
        return dict(job) if job else None


def set_job(job_id: str, job_data: dict) -> None:
    with _JOBS_CACHE_LOCK:
        JOBS_CACHE[str(job_id)] = dict(job_data)


def update_job(job_id: str, **changes) -> dict:
    with _JOBS_CACHE_LOCK:
        current = dict(JOBS_CACHE.get(job_id, {}))
        current.update(changes)
        current["job_id"] = job_id
        current["updated_at"] = datetime.now().isoformat()

        JOBS_CACHE[job_id] = current
        return dict(current)


def delete_job(job_id: str) -> None:
    with _JOBS_CACHE_LOCK:
        JOBS_CACHE.pop(job_id, None)


def recover_interrupted_jobs() -> None:
    """
    Marca jobs que estavam rodando antes de um restart como aguardando.

    A retomada real será feita pela parte do pipeline que verifica o
    checkpoint resume.pt.
    """
    active_statuses = {
        STATUS_WAITING,
        STATUS_PREPROCESSING,
        STATUS_EXTRACTING_F0,
        STATUS_EXTRACTING_FEATURES,
        STATUS_TRAINING_INDEX,
        STATUS_TRAINING,
    }

    with _JOBS_CACHE_LOCK:
        for job in JOBS_CACHE.values():
            if job.get("status") in active_statuses:
                job["status"] = STATUS_WAITING
                job["message"] = (
                    "Space reiniciada. Job aguardando retomada pelo checkpoint."
                )
                job["recovered"] = True
                job["updated_at"] = datetime.now().isoformat()


_init_cache()
recover_interrupted_jobs()

_disk_writer_thread = threading.Thread(
    target=_disk_writer_daemon,
    name="rvc-jobs-disk-writer",
    daemon=True,
)
_disk_writer_thread.start()

# ============================================================================
# LOGS POR JOB
# ============================================================================

_LOG_BUFFERS: Dict[str, List[str]] = {}
_LOG_BUFFER_LOCK = threading.RLock()


def append_log(job_id: str, message: str) -> None:
    if not job_id:
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}\n"

    with _LOG_BUFFER_LOCK:
        buffer = _LOG_BUFFERS.setdefault(job_id, [])
        buffer.append(line)

        should_flush = (
            len(buffer) >= 10
            or "ERRO" in message.upper()
            or "FATAL" in message.upper()
        )

        if not should_flush:
            return

        lines = "".join(buffer)
        _LOG_BUFFERS[job_id] = []

    _write_job_log(job_id, lines)


def _write_job_log(job_id: str, lines: str) -> None:
    if not lines:
        return

    path = LOGS_DIR / f"{job_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(lines)


def flush_all_logs() -> None:
    with _LOG_BUFFER_LOCK:
        buffers = dict(_LOG_BUFFERS)
        _LOG_BUFFERS.clear()

    for job_id, lines in buffers.items():
        _write_job_log(job_id, "".join(lines))


# ============================================================================
# UTILITÁRIOS
# ============================================================================

def generate_job_id() -> str:
    return (
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{random.randint(1000, 9999)}"
    )


def sanitize_filename(name: str) -> str:
    if not name:
        return "modelo_rvc"

    value = re.sub(
        r"[^a-zA-Z0-9_\-]",
        "_",
        str(name).strip(),
    )

    value = re.sub(r"_+", "_", value).strip("_")
    return value[:50] or "modelo_rvc"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "N/A"

    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds}s"

    minutes, remaining_seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {remaining_seconds}s"


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


FFMPEG_AVAILABLE = check_ffmpeg()

if not FFMPEG_AVAILABLE:
    logger.warning("ffmpeg não encontrado.")


def get_media_duration(filepath: str) -> Optional[float]:
    """
    Obtém duração usando soundfile/librosa antes de tentar ffprobe.
    """
    try:
        info = sf.info(filepath)
        if info.samplerate > 0:
            return float(info.frames / info.samplerate)
    except Exception:
        pass

    try:
        duration = librosa.get_duration(path=filepath)
        return float(duration)
    except Exception:
        pass

    return None


def convert_audio_to_wav(
    input_path: str,
    output_path: str,
    sample_rate: int = SR_TARGET,
) -> str:
    if not FFMPEG_AVAILABLE:
        raise RuntimeError(
            "ffmpeg não está instalado na Space."
        )

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-af",
        "highpass=f=80,lowpass=f=8000,afftdn=nf=-25",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg falhou: {result.stderr[-1000:]}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produziu um arquivo vazio.")

    return str(output_path)


def extract_audio_from_video(
    video_path: str,
    output_path: str,
    sample_rate: int = SR_TARGET,
) -> str:
    return convert_audio_to_wav(
        video_path,
        output_path,
        sample_rate,
    )


# ============================================================================
# PIPELINE INICIAL DE ÁUDIO
# ============================================================================

def process_audio_pipeline(
    input_path: str,
    output_dir: str,
    sample_rate: int = SR_TARGET,
) -> dict:
    """
    Converte o áudio para WAV mono, aplica filtros e cria segmentos.

    A duração dos segmentos é calculada em memória. Não chama ffprobe uma
    vez por segmento.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Arquivo de áudio não encontrado: {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    temporary_wav = output_dir / "input_converted.wav"
    processed_wav = output_dir / "processed.wav"

    convert_audio_to_wav(
        input_path,
        temporary_wav,
        sample_rate,
    )

    filter_chain = (
        "silenceremove=stop_periods=-1:"
        "stop_duration=0.5:"
        "stop_threshold=-50dB,"
        "loudnorm=I=-16:LRA=11:TP=-1.5,"
        "afftdn=nf=-25,"
        "highpass=f=80,"
        "lowpass=f=8000,"
        "alimiter=level_in=1:level_out=1:attack=5:release=50"
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(temporary_wav),
        "-af",
        filter_chain,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        str(processed_wav),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    if (
        result.returncode != 0
        or not processed_wav.exists()
        or processed_wav.stat().st_size == 0
    ):
        logger.warning(
            "Filtro avançado falhou; usando áudio convertido."
        )
        shutil.copy2(temporary_wav, processed_wav)

    segments = []
    total_duration = 0.0

    try:
        audio, actual_sr = librosa.load(
            str(processed_wav),
            sr=int(sample_rate),
            mono=True,
        )

        segment_duration = 3.7
        overlap = 0.3

        segment_length = int(segment_duration * actual_sr)
        hop_length = int(
            (segment_duration - overlap) * actual_sr
        )

        if len(audio) < segment_length:
            audio = np.pad(
                audio,
                (0, segment_length - len(audio)),
            )

        segment_index = 0

        for start in range(
            0,
            max(1, len(audio) - segment_length + 1),
            hop_length,
        ):
            segment = audio[start:start + segment_length]

            if len(segment) < segment_length:
                segment = np.pad(
                    segment,
                    (0, segment_length - len(segment)),
                )

            rms = float(np.sqrt(np.mean(segment ** 2)))

            if not np.isfinite(rms) or rms <= 0.001:
                continue

            segment_path = output_dir / (
                f"segment_{segment_index:04d}.wav"
            )

            clipped = np.clip(segment, -1.0, 1.0)
            wavfile.write(
                str(segment_path),
                int(actual_sr),
                (clipped * 32767.0).astype(np.int16),
            )

            segments.append(str(segment_path))
            total_duration += len(segment) / actual_sr
            segment_index += 1

        if not segments:
            raise RuntimeError(
                "Nenhum segmento com volume suficiente foi encontrado."
            )

    except Exception as exc:
        logger.warning(
            "Segmentação falhou: %s",
            exc,
        )

        fallback_path = output_dir / "segment_0000.wav"
        shutil.copy2(processed_wav, fallback_path)

        segments = [str(fallback_path)]
        total_duration = get_media_duration(str(processed_wav)) or 0.0

    temporary_wav.unlink(missing_ok=True)

    return {
        "segments": segments,
        "total_segments": len(segments),
        "total_duration": total_duration,
        "sample_rate": int(sample_rate),
        "processed_file": str(processed_wav),
    }


# ============================================================================
# SLICER RVC
# ============================================================================

def get_rms(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if y.ndim != 1:
        y = np.mean(y, axis=0)

    frame_length = max(1, int(frame_length))
    hop_length = max(1, int(hop_length))

    padding = frame_length // 2
    padded = np.pad(
        y,
        (padding, padding),
        mode="constant",
    )

    if len(padded) < frame_length:
        padded = np.pad(
            padded,
            (0, frame_length - len(padded)),
        )

    frames = np.lib.stride_tricks.sliding_window_view(
        padded,
        frame_length,
    )[::hop_length]

    return np.sqrt(
        np.mean(np.square(frames), axis=-1)
    )


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        self.sr = int(sr)
        self.threshold = 10 ** (threshold / 20.0)

        self.hop_size = max(
            1,
            round(self.sr * hop_size / 1000),
        )

        min_interval_samples = self.sr * min_interval / 1000
        self.win_size = max(
            1,
            min(
                round(min_interval_samples),
                4 * self.hop_size,
            ),
        )

        self.min_length = max(
            1,
            round(
                self.sr
                * min_length
                / 1000
                / self.hop_size
            ),
        )

        self.min_interval = max(
            1,
            round(
                min_interval_samples
                / self.hop_size
            ),
        )

        self.max_sil_kept = max(
            1,
            round(
                self.sr
                * max_sil_kept
                / 1000
                / self.hop_size
            ),
        )

    def _apply_slice(
        self,
        waveform: np.ndarray,
        begin: int,
        end: int,
    ) -> np.ndarray:
        begin_sample = max(0, int(begin * self.hop_size))
        end_sample = max(begin_sample, int(end * self.hop_size))

        if waveform.ndim > 1:
            end_sample = min(waveform.shape[1], end_sample)
            return waveform[:, begin_sample:end_sample]

        end_sample = min(waveform.shape[0], end_sample)
        return waveform[begin_sample:end_sample]

    def slice(self, waveform: np.ndarray) -> List[np.ndarray]:
        waveform = np.asarray(waveform)

        samples = (
            waveform.mean(axis=0)
            if waveform.ndim > 1
            else waveform
        )

        if samples.size <= self.min_length:
            return [waveform]

        rms_list = get_rms(
            samples,
            frame_length=self.win_size,
            hop_length=self.hop_size,
        )

        silence_tags = []
        silence_start = None
        clip_start = 0

        for index, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = index
                continue

            if silence_start is None:
                continue

            is_leading_silence = (
                silence_start == 0
                and index > self.max_sil_kept
            )

            need_middle_slice = (
                index - silence_start >= self.min_interval
                and index - clip_start >= self.min_length
            )

            if not is_leading_silence and not need_middle_slice:
                silence_start = None
                continue

            if index - silence_start <= self.max_sil_kept:
                local = rms_list[silence_start:index + 1]
                position = int(
                    np.argmin(local) + silence_start
                )

                if silence_start == 0:
                    silence_tags.append((0, position))
                else:
                    silence_tags.append((position, position))

                clip_start = position

            elif index - silence_start <= self.max_sil_kept * 2:
                left_region = rms_list[
                    silence_start:
                    silence_start + self.max_sil_kept + 1
                ]

                right_region = rms_list[
                    index - self.max_sil_kept:
                    index + 1
                ]

                left = int(
                    np.argmin(left_region) + silence_start
                )

                right = int(
                    np.argmin(right_region)
                    + index
                    - self.max_sil_kept
                )

                if silence_start == 0:
                    silence_tags.append((0, right))
                    clip_start = right
                else:
                    silence_tags.append(
                        (min(left, right), max(left, right))
                    )
                    clip_start = max(left, right)

            else:
                left_region = rms_list[
                    silence_start:
                    silence_start + self.max_sil_kept + 1
                ]

                right_region = rms_list[
                    index - self.max_sil_kept:
                    index + 1
                ]

                left = int(
                    np.argmin(left_region) + silence_start
                )

                right = int(
                    np.argmin(right_region)
                    + index
                    - self.max_sil_kept
                )

                if silence_start == 0:
                    silence_tags.append((0, right))
                else:
                    silence_tags.append((left, right))

                clip_start = right

            silence_start = None

        total_frames = len(rms_list)

        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(
                total_frames,
                silence_start + self.max_sil_kept,
            )

            local = rms_list[
                silence_start:silence_end + 1
            ]

            position = int(
                np.argmin(local) + silence_start
            )

            silence_tags.append(
                (position, total_frames + 1)
            )

        if not silence_tags:
            return [waveform]

        chunks = []

        if silence_tags[0][0] > 0:
            chunks.append(
                self._apply_slice(
                    waveform,
                    0,
                    silence_tags[0][0],
                )
            )

        for index in range(len(silence_tags) - 1):
            chunks.append(
                self._apply_slice(
                    waveform,
                    silence_tags[index][1],
                    silence_tags[index + 1][0],
                )
            )

        if silence_tags[-1][1] < total_frames:
            chunks.append(
                self._apply_slice(
                    waveform,
                    silence_tags[-1][1],
                    total_frames,
                )
            )

        return [
            chunk for chunk in chunks
            if chunk is not None and chunk.size > 0
        ]


# ============================================================================
# PRÉ-PROCESSAMENTO RVC
# ============================================================================

def load_audio(path: str, sr: int) -> np.ndarray:
    audio, _ = librosa.load(
        path,
        sr=int(sr),
        mono=True,
    )

    return audio.astype(np.float32)


class PreProcess:
    def __init__(
        self,
        sr: int,
        exp_dir: str,
        per: float = 3.7,
    ):
        self.sr = int(sr)
        self.exp_dir = Path(exp_dir)
        self.per = float(per)
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max_amplitude = 0.9
        self.alpha = 0.75

        self.slicer = Slicer(
            sr=self.sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )

        self.bh, self.ah = signal.butter(
            N=5,
            Wn=48,
            btype="high",
            fs=self.sr,
        )

        self.gt_wavs_dir = self.exp_dir / "0_gt_wavs"
        self.wavs16k_dir = self.exp_dir / "1_16k_wavs"

        self.gt_wavs_dir.mkdir(parents=True, exist_ok=True)
        self.wavs16k_dir.mkdir(parents=True, exist_ok=True)

        self.n_written = 0
        self.n_discarded = 0

    def norm_write(
        self,
        audio: np.ndarray,
        index_0: int,
        index_1: int,
    ) -> bool:
        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            self.n_discarded += 1
            return False

        maximum = float(np.max(np.abs(audio)))

        if (
            not np.isfinite(maximum)
            or maximum <= 0
            or maximum > 2.5
        ):
            self.n_discarded += 1
            return False

        audio = (
            audio / maximum * self.max_amplitude * self.alpha
            + (1.0 - self.alpha) * audio
        )

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        audio = np.clip(audio, -1.0, 1.0)

        name = f"{index_0}_{index_1}.wav"

        gt_path = self.gt_wavs_dir / name
        wav16k_path = self.wavs16k_dir / name

        wavfile.write(
            str(gt_path),
            self.sr,
            audio.astype(np.float32),
        )

        audio_16k = librosa.resample(
            audio,
            orig_sr=self.sr,
            target_sr=16000,
        ).astype(np.float32)

        wavfile.write(
            str(wav16k_path),
            16000,
            audio_16k,
        )

        self.n_written += 1
        return True

    def pipeline(
        self,
        path: str,
        index_0: int,
        total: int,
    ) -> bool:
        try:
            audio = load_audio(path, self.sr)
            audio = signal.lfilter(
                self.bh,
                self.ah,
                audio,
            )

            for audio_slice in self.slicer.slice(audio):
                segment_index = 0
                start_index = 0

                while True:
                    start = int(
                        self.sr
                        * (self.per - self.overlap)
                        * start_index
                    )

                    start_index += 1
                    remaining = audio_slice[start:]

                    if len(remaining) > self.tail * self.sr:
                        current = audio_slice[
                            start:
                            start + int(self.per * self.sr)
                        ]

                        self.norm_write(
                            current,
                            index_0,
                            segment_index,
                        )

                        segment_index += 1
                    else:
                        self.norm_write(
                            remaining,
                            index_0,
                            segment_index,
                        )
                        break

            return True

        except Exception as exc:
            logger.warning(
                "Erro no pré-processamento de %s: %s",
                path,
                exc,
            )
            return False


def preprocess_trainset(
    input_root: str,
    sr: int,
    exp_dir: str,
    per: float = 3.7,
) -> dict:
    input_root = Path(input_root)

    names = sorted(
        path.name
        for path in input_root.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
    )

    processor = PreProcess(
        sr=sr,
        exp_dir=exp_dir,
        per=per,
    )

    successful = 0

    for index, name in enumerate(names):
        if processor.pipeline(
            str(input_root / name),
            index,
            len(names),
        ):
            successful += 1

    if processor.n_discarded:
        logger.warning(
            "Pré-processamento: %d segmentos salvos, "
            "%d descartados.",
            processor.n_written,
            processor.n_discarded,
        )

    return {
        "input_files": len(names),
        "successful_files": successful,
        "written_segments": processor.n_written,
        "discarded_segments": processor.n_discarded,
    }


# ============================================================================
# EXTRAÇÃO DE F0
# ============================================================================

class FeatureInput:
    def __init__(
        self,
        samplerate: int = 16000,
        hop_size: int = 160,
    ):
        self.fs = int(samplerate)
        self.hop = int(hop_size)

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0

        self.f0_mel_min = (
            1127
            * np.log(1 + self.f0_min / 700)
        )

        self.f0_mel_max = (
            1127
            * np.log(1 + self.f0_max / 700)
        )

    def compute_f0(self, path: str) -> np.ndarray:
        audio = load_audio(path, self.fs)

        expected_length = max(
            1,
            audio.shape[0] // self.hop,
        )

        try:
            f0 = librosa.yin(
                audio,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=self.fs,
                hop_length=self.hop,
            )
        except Exception:
            f0 = np.zeros(expected_length, dtype=np.float32)

        f0 = np.asarray(f0, dtype=np.float32)

        f0 = np.nan_to_num(
            f0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if len(f0) < expected_length:
            f0 = np.pad(
                f0,
                (0, expected_length - len(f0)),
            )
        else:
            f0 = f0[:expected_length]

        voiced = f0 > 0
        unvoiced = ~voiced

        if np.any(voiced) and np.any(unvoiced):
            voiced_indices = np.where(voiced)[0]
            unvoiced_indices = np.where(unvoiced)[0]

            f0[unvoiced_indices] = np.interp(
                unvoiced_indices,
                voiced_indices,
                f0[voiced_indices],
            )

        return f0.astype(np.float32)

    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        f0 = np.asarray(f0, dtype=np.float32)

        f0_mel = 1127 * np.log1p(f0 / 700)

        positive = f0_mel > 0

        f0_mel[positive] = (
            (
                f0_mel[positive]
                - self.f0_mel_min
            )
            * (self.f0_bin - 2)
            / (
                self.f0_mel_max
                - self.f0_mel_min
            )
            + 1
        )

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = (
            self.f0_bin - 1
        )

        return np.rint(f0_mel).astype(np.int64)


def run_f0_extraction(exp_dir: str) -> int:
    feature_input = FeatureInput()

    input_root = Path(exp_dir) / "1_16k_wavs"
    coarse_root = Path(exp_dir) / "2a_f0"
    fine_root = Path(exp_dir) / "2b-f0nsf"

    coarse_root.mkdir(parents=True, exist_ok=True)
    fine_root.mkdir(parents=True, exist_ok=True)

    names = sorted(
        path.name
        for path in input_root.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
    )

    for name in names:
        base_name = Path(name).stem
        input_path = input_root / name

        f0 = feature_input.compute_f0(str(input_path))
        coarse = feature_input.coarse_f0(f0)

        np.save(
            fine_root / f"{base_name}.npy",
            f0,
            allow_pickle=False,
        )

        np.save(
            coarse_root / f"{base_name}.npy",
            coarse,
            allow_pickle=False,
        )

    return len(names)
    # ============================================================================
# EXTRAÇÃO DE FEATURES COM HUBERT
# ============================================================================

def run_feature_extraction(
    exp_dir: str,
    version: str = "v2",
    job_id: Optional[str] = None,
) -> Tuple[int, bool]:
    """
    Extrai features usando HuBERT real (lj1995/VoiceConversionWebUI/hubert_base).

    v1 usa 256-dim (projeção linear fixa de 768 -> 256).
    v2 usa 768-dim (saída direta do HuBERT).

    Se o modelo não puder ser carregado, usa fallback de projeção aleatória
    (apenas ruído estruturado, sem conteúdo fonético real).

    Retorna: (num_files, used_real_hubert)
    """
    input_root = Path(exp_dir) / "1_16k_wavs"
    output_root = (
        Path(exp_dir) / "3_feature256"
        if version == "v1"
        else Path(exp_dir) / "3_feature768"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    feat_dim = 256 if version == "v1" else 768

    hubert_model = None
    used_real_hubert = False

    try:
        from transformers import HubertModel

        hubert_cache_dir = str(ASSETS_DIR / "hubert_base")

        required_files = {
            "pytorch_model.bin",
            "config.json",
            "preprocessor_config.json",
        }

        cache_valid = (
            Path(hubert_cache_dir).exists()
            and all(
                (Path(hubert_cache_dir) / f).exists()
                for f in required_files
            )
        )

        if not cache_valid:
            logger.info(
                "Baixando HuBERT (lj1995/VoiceConversionWebUI)..."
            )

            if job_id:
                append_log(
                    job_id,
                    "Baixando content-vec (HuBERT)...",
                )

            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id="lj1995/VoiceConversionWebUI",
                    allow_patterns=["hubert_base/*"],
                    local_dir=str(ASSETS_DIR),
                    cache_dir=None,
                )
            except Exception as download_exc:
                logger.warning(
                    "Falha ao baixar HuBERT: %s",
                    download_exc,
                )

                if job_id:
                    append_log(
                        job_id,
                        f"AVISO: Não foi possível baixar HuBERT. "
                        f"Usando fallback de projeção aleatória: {download_exc}",
                    )

                raise download_exc

        missing = [
            f
            for f in required_files
            if not (Path(hubert_cache_dir) / f).exists()
        ]

        if missing:
            raise RuntimeError(
                f"Download incompleto, faltando: {missing}"
            )

        hubert_model = HubertModel.from_pretrained(
            hubert_cache_dir,
            torch_dtype=torch.float32,
        )

        hubert_model = hubert_model.to(DEVICE).eval()
        used_real_hubert = True

        logger.info(
            "HuBERT real carregado de %s (768-dim)",
            hubert_cache_dir,
        )

        if job_id:
            append_log(
                job_id,
                "Content-vec (HuBERT) carregado com sucesso.",
            )

    except Exception as exc:
        logger.warning(
            "Falha ao carregar HuBERT, usando fallback: %s",
            exc,
        )

        if job_id:
            append_log(
                job_id,
                f"AVISO: Não foi possível carregar HuBERT. "
                f"Features serão geradas via fallback (projeção aleatória, "
                f"sem conteúdo fonético real). Qualidade do treinamento será "
                f"severamente reduzida: {exc}",
            )

        hubert_model = None
        used_real_hubert = False

    v1_projection = None

    if version == "v1" and used_real_hubert:
        rng = np.random.RandomState(42)
        v1_projection = (
            rng.randn(768, 256).astype(np.float32)
            * (1.0 / np.sqrt(768))
        )

    names = sorted(
        path.name
        for path in input_root.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
    )

    for name in names:
        base_name = Path(name).stem
        output_file = output_root / f"{base_name}.npy"

        if output_file.exists():
            continue

        input_path = input_root / name

        try:
            if hubert_model is not None:
                audio, sr = librosa.load(
                    str(input_path),
                    sr=16000,
                    mono=True,
                )

                audio_tensor = (
                    torch.from_numpy(audio)
                    .float()
                    .unsqueeze(0)
                    .to(DEVICE)
                    .to(torch.float32)
                )

                with torch.no_grad():
                    output = hubert_model(audio_tensor)
                    features = (
                        output.last_hidden_state
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )

                if version == "v1" and v1_projection is not None:
                    features = (
                        features @ v1_projection
                    ).astype(np.float32)

            else:
                audio, sr = librosa.load(
                    str(input_path),
                    sr=16000,
                    mono=True,
                )

                n_frames = max(1, len(audio) // 320)

                stft = np.abs(
                    librosa.stft(
                        audio,
                        n_fft=512,
                        hop_length=320,
                    )
                )

                if stft.shape[1] < n_frames:
                    stft = np.pad(
                        stft,
                        ((0, 0), (0, n_frames - stft.shape[1])),
                    )
                else:
                    stft = stft[:, :n_frames]

                n_bins = stft.shape[0]

                rng = np.random.RandomState(42)
                projection = (
                    rng.randn(feat_dim, n_bins)
                    .astype(np.float32)
                    * 0.01
                )

                features = (projection @ stft).T.astype(
                    np.float32
                )

                features = features / (
                    np.linalg.norm(
                        features,
                        axis=1,
                        keepdims=True,
                    )
                    + 1e-8
                ) * 0.5

            if features.shape[1] != feat_dim:
                raise RuntimeError(
                    f"Dimensão de feature incorreta para {name}: "
                    f"{features.shape[1]} != {feat_dim}"
                )

            np.save(
                output_file,
                features,
                allow_pickle=False,
            )

        except Exception as exc:
            logger.exception(
                "Erro ao extrair feature de %s",
                name,
            )

            if job_id:
                append_log(
                    job_id,
                    f"ERRO ao extrair feature de {name}: {exc}",
                )

    return len(names), used_real_hubert


# ============================================================================
# ÍNDICE FAISS
# ============================================================================

def train_index(
    exp_name: str,
    version: str = "v2",
    exp_dir_base: str = "logs",
) -> Optional[str]:
    """
    Treina um índice FAISS com as features extraídas.
    Retorna o caminho do arquivo .index ou None.
    """
    try:
        import faiss
    except ImportError:
        logger.warning("faiss não instalado. Pulando índice.")
        return None

    exp_dir = Path(exp_dir_base) / exp_name
    feature_dir = (
        exp_dir / "3_feature256"
        if version == "v1"
        else exp_dir / "3_feature768"
    )

    if not feature_dir.exists() or not list(feature_dir.glob("*.npy")):
        logger.warning("Nenhuma feature encontrada para treinar FAISS em %s", exp_dir)
        return None

    try:
        features = []

        for npy_file in sorted(feature_dir.glob("*.npy")):
            try:
                feat = np.load(npy_file)
                if feat.size > 0:
                    if feat.ndim == 1:
                        feat = feat.reshape(1, -1)
                    features.append(feat.astype(np.float32))
            except Exception:
                logger.warning("Erro ao carregar %s", npy_file)

        if not features:
            logger.warning("Nenhuma feature válida foi encontrada para FAISS.")
            return None

        big_npy = np.concatenate(features, axis=0)
        big_npy = big_npy[np.random.permutation(big_npy.shape[0])]

        dim = 256 if version == "v1" else 768
        n_vectors = big_npy.shape[0]

        min_ivf_points = 39 * 4

        if n_vectors < min_ivf_points:
            logger.info(
                "Poucos vetores (%d < %d). Usando IndexFlatL2.",
                n_vectors,
                min_ivf_points,
            )
            index = faiss.IndexFlatL2(dim)
            index.add(big_npy.astype(np.float32))
            index_path = exp_dir / f"added_Flat_{exp_name}_{version}.index"
        else:
            n_ivf = max(
                1,
                min(
                    int(16 * np.sqrt(n_vectors)),
                    n_vectors // 39,
                ),
            )

            index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
            index.train(big_npy.astype(np.float32))
            index.add(big_npy.astype(np.float32))
            index_path = exp_dir / f"added_IVF{n_ivf}_Flat_{exp_name}_{version}.index"

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

        logger.info("Índice FAISS salvo em %s", index_path)
        return str(index_path)

    except Exception as exc:
        logger.exception("Falha ao treinar índice FAISS: %s", exc)
        return None
        
# ============================================================================
# FUNÇÕES AUXILIARES DO MODELO
# ============================================================================

def commons_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None,
) -> torch.Tensor:
    if max_len is None:
        max_len = int(torch.max(lengths).item())

    batch_size = lengths.size(0)

    x = torch.arange(
        max_len,
        dtype=lengths.dtype,
        device=lengths.device,
    )

    return x.unsqueeze(0) < lengths.unsqueeze(1)


def commons_slice_segments(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int,
) -> torch.Tensor:
    """
    Fatia segmentos fixos do tensor x usando índices em ids_str.

    Se x for mais curto que segment_size, faz padding com zeros.
    """
    batch_size, channels, length = x.size()

    if length < segment_size:
        pad_amount = segment_size - length
        x = F.pad(x, (0, pad_amount))
        length = segment_size

    ret = torch.zeros(
        batch_size,
        channels,
        segment_size,
        dtype=x.dtype,
        device=x.device,
    )

    for i in range(batch_size):
        idx = int(ids_str[i].item())
        idx = max(0, min(idx, length - segment_size))

        ret[i] = x[i, :, idx:idx + segment_size]

    return ret


def commons_rand_slice(
    x: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    segment_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fatia aleatória com clamp seguro para amostras curtas.

    x_lengths pode ser um tensor (spec_lengths) com comprimentos por amostra.
    """
    batch_size, channels, length = x.size()

    if x_lengths is None:
        x_lengths = torch.full(
            (batch_size,),
            length,
            dtype=torch.long,
            device=x.device,
        )

    elif not torch.is_tensor(x_lengths):
        x_lengths = torch.full(
            (batch_size,),
            int(x_lengths),
            dtype=torch.long,
            device=x.device,
        )

    else:
        x_lengths = x_lengths.to(x.device)

    # Garante que nunca pedimos um segmento maior que a amostra
    safe_lengths = torch.clamp(
        x_lengths,
        min=segment_size,
    )

    ids_str_max = torch.clamp(
        safe_lengths - segment_size + 1,
        min=1,
    )

    ids_str = (
        torch.rand(
            batch_size,
            device=x.device,
        )
        * ids_str_max.float()
    ).to(dtype=torch.long)

    ids_str = torch.clamp(
        ids_str,
        min=0,
        max=max(0, length - segment_size),
    )

    return commons_slice_segments(
        x,
        ids_str,
        segment_size,
    ), ids_str


# ============================================================================
# CAMADAS DO MODELO RVC
# ============================================================================

class LayerNorm(nn.Module):
    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)

        x = F.layer_norm(
            x,
            (self.channels,),
            self.gamma,
            self.beta,
            self.eps,
        )

        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        heads_share: bool = True,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()

        assert channels % n_heads == 0, (
            "channels deve ser divisível por n_heads"
        )

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.k_channels = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)

        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads

            rel_stddev = self.k_channels ** -0.5

            self.emb_rel_k = nn.Parameter(
                torch.randn(
                    n_heads_rel,
                    window_size * 2 + 1,
                    self.k_channels,
                )
                * rel_stddev
            )

            self.emb_rel_v = nn.Parameter(
                torch.randn(
                    n_heads_rel,
                    window_size * 2 + 1,
                    self.k_channels,
                )
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                if self.conv_k.bias is not None:
                    self.conv_k.bias.copy_(
                        self.conv_q.bias
                    )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, _ = self.attention(
            q,
            k,
            v,
            mask=attn_mask,
        )

        x = self.conv_o(x)

        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, dim, t_s = key.size()
        t_t = query.size(2)

        query = query.view(
            batch_size,
            self.n_heads,
            self.k_channels,
            t_t,
        ).transpose(2, 3)

        key = key.view(
            batch_size,
            self.n_heads,
            self.k_channels,
            t_s,
        ).transpose(2, 3)

        value = value.view(
            batch_size,
            self.n_heads,
            self.k_channels,
            t_s,
        ).transpose(2, 3)

        scores = torch.matmul(
            query / math.sqrt(self.k_channels),
            key.transpose(-2, -1),
        )

        if self.window_size is not None:
            key_relative_embeddings = (
                self._get_relative_embeddings(
                    self.emb_rel_k,
                    t_s,
                )
            )

            rel_logits = (
                self._matmul_with_relative_keys(
                    query / math.sqrt(self.k_channels),
                    key_relative_embeddings,
                )
            )

            scores_local = (
                self._relative_position_to_absolute_position(
                    rel_logits
                )
            )

            scores = scores + scores_local

        if self.proximal_bias:
            scores = scores + self._attention_bias_proximal(
                t_s
            ).to(device=scores.device, dtype=scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)

        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = (
                self._absolute_position_to_relative_position(
                    p_attn
                )
            )

            value_relative_embeddings = (
                self._get_relative_embeddings(
                    self.emb_rel_v,
                    t_s,
                )
            )

            output = output + (
                self._matmul_with_relative_values(
                    relative_weights,
                    value_relative_embeddings,
                )
            )

        output = output.transpose(2, 3).contiguous().view(
            batch_size,
            dim,
            t_t,
        )

        return output, p_attn

    def _matmul_with_relative_values(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(
            x,
            y.unsqueeze(0).transpose(-2, -1),
        )

    def _get_relative_embeddings(
        self,
        relative_embeddings: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        pad_length = max(
            length - (self.window_size + 1),
            0,
        )

        slice_start_position = max(
            (self.window_size + 1) - length,
            0,
        )

        slice_end_position = (
            slice_start_position + 2 * length - 1
        )

        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                [0, 0, pad_length, pad_length, 0, 0],
            )
        else:
            padded_relative_embeddings = (
                relative_embeddings
            )

        return padded_relative_embeddings[
            :,
            slice_start_position:slice_end_position,
        ]

    def _relative_position_to_absolute_position(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, length, _ = x.size()

        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])

        x_flat = x.view(
            [batch, heads, length * 2 * length]
        )

        x_flat = F.pad(
            x_flat,
            [0, int(length) - 1, 0, 0, 0, 0],
        )

        x_final = x_flat.view([
            batch,
            heads,
            length + 1,
            2 * length - 1,
        ])[:, :, :length, length - 1:]

        return x_final

    def _absolute_position_to_relative_position(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, length, _ = x.size()

        x = F.pad(
            x,
            [0, int(length) - 1, 0, 0, 0, 0, 0, 0],
        )

        x_flat = x.view([
            batch,
            heads,
            int(length ** 2) + int(length * (length - 1)),
        ])

        x_flat = F.pad(
            x_flat,
            [length, 0, 0, 0, 0, 0],
        )

        x_final = x_flat.view([
            batch,
            heads,
            length,
            2 * length,
        ])[:, :, :, 1:]

        return x_final

    def _attention_bias_proximal(
        self,
        length: int,
    ) -> torch.Tensor:
        r = torch.arange(
            length,
            dtype=torch.float32,
        )

        diff = (
            torch.unsqueeze(r, 0)
            - torch.unsqueeze(r, 1)
        )

        return torch.unsqueeze(
            torch.unsqueeze(
                -torch.log1p(torch.abs(diff)),
                0,
            ),
            0,
        )


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: Optional[str] = None,
        causal: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.causal = causal

        self.is_activation = activation == "gelu"

        self.conv_1 = nn.Conv1d(
            in_channels,
            filter_channels,
            kernel_size,
        )

        self.conv_2 = nn.Conv1d(
            filter_channels,
            out_channels,
            kernel_size,
        )

        self.drop = nn.Dropout(p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv_1(self._padding(x, x_mask))

        if self.is_activation:
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)

        x = self.drop(x)
        x = self.conv_2(self._padding(x, x_mask))

        return x * x_mask

    def _padding(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.causal:
            if self.kernel_size == 1:
                return x * x_mask

            pad_l = self.kernel_size - 1
            return F.pad(
                x * x_mask,
                [pad_l, 0, 0, 0, 0, 0],
            )

        else:
            if self.kernel_size == 1:
                return x * x_mask

            pad_l = (self.kernel_size - 1) // 2
            pad_r = self.kernel_size // 2

            return F.pad(
                x * x_mask,
                [pad_l, pad_r, 0, 0, 0, 0],
            )


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
        **kwargs,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.n_layers = int(n_layers)

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )

            self.norm_layers_1.append(
                LayerNorm(hidden_channels)
            )

            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )

            self.norm_layers_2.append(
                LayerNorm(hidden_channels)
            )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = (
            x_mask.unsqueeze(2)
            * x_mask.unsqueeze(-1)
        )

        x = x * x_mask

        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers,
            self.norm_layers_1,
            self.ffn_layers,
            self.norm_layers_2,
        ):
            y = attn(x, x, attn_mask)
            y = self.drop(y)
            x = norm1(x + y)

            y = ffn(x, x_mask)
            y = self.drop(y)
            x = norm2(x + y)

        return x * x_mask


class TextEncoder768(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb_phone = nn.Linear(768, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)

        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )

        self.proj = nn.Conv1d(
            hidden_channels,
            out_channels * 2,
            1,
        )

    def forward(
        self,
        phone: torch.Tensor,
        pitch: Optional[torch.Tensor],
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phone: (batch, time, 768)
            pitch: (batch, time) ou None
            lengths: (batch,)

        Returns:
            m: (batch, out_channels, time)
            logs: (batch, out_channels, time)
            x_mask: (batch, 1, time)
        """
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = (
                self.emb_phone(phone)
                + self.emb_pitch(pitch)
            )

        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)

        x = torch.transpose(x, 1, -1)

        x_mask = torch.unsqueeze(
            commons_mask(lengths, x.size(2)),
            1,
        ).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)

        stats = self.proj(x) * x_mask

        m, logs = torch.split(
            stats,
            self.out_channels,
            dim=1,
        )

        return m, logs, x_mask


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
    ):
        super().__init__()

        assert kernel_size % 2 == 1

        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = float(p_dropout)

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(float(p_dropout))

        if gin_channels != 0:
            cond_layer = nn.Conv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                1,
            )

            self.cond_layer = nn.utils.weight_norm(
                cond_layer,
                name="weight",
            )

        for i in range(n_layers):
            dilation = dilation_rate ** i

            padding = int(
                (kernel_size * dilation - dilation) / 2
            )

            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )

            in_layer = nn.utils.weight_norm(
                in_layer,
                name="weight",
            )

            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(
                hidden_channels,
                res_skip_channels,
                1,
            )

            res_skip_layer = nn.utils.weight_norm(
                res_skip_layer,
                name="weight",
            )

            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(
            zip(self.in_layers, self.res_skip_layers)
        ):
            x_in = in_layer(x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels

                g_l = g[
                    :,
                    cond_offset:cond_offset
                    + 2 * self.hidden_channels,
                    :,
                ]
            else:
                g_l = torch.zeros_like(x_in)

            acts = (
                torch.tanh(
                    x_in[:, :self.hidden_channels, :]
                    + g_l[:, :self.hidden_channels, :]
                )
                * torch.sigmoid(
                    x_in[:, self.hidden_channels:, :]
                    + g_l[:, self.hidden_channels:, :]
                )
            )

            acts = self.drop(acts)

            res_skip_acts = res_skip_layer(acts)

            if i < self.n_layers - 1:
                res_acts = res_skip_acts[
                    :,
                    :self.hidden_channels,
                    :,
                ]

                x = (x + res_acts) * x_mask

                output = output + res_skip_acts[
                    :,
                    self.hidden_channels:,
                    :,
                ]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            nn.utils.remove_weight_norm(
                self.cond_layer
            )

        for layer in self.in_layers:
            nn.utils.remove_weight_norm(layer)

        for layer in self.res_skip_layers:
            nn.utils.remove_weight_norm(layer)


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        assert channels % 2 == 0

        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(
            self.half_channels,
            hidden_channels,
            1,
        )

        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=float(p_dropout),
            gin_channels=gin_channels,
        )

        self.post = nn.Conv1d(
            hidden_channels,
            self.half_channels * (2 - mean_only),
            1,
        )

        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.split(
            x,
            [self.half_channels] * 2,
            1,
        )

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)

        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(
                stats,
                [self.half_channels] * 2,
                1,
            )
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = (
                m
                + x1 * torch.exp(logs) * x_mask
            )

            x = torch.cat([x0, x1], 1)

            logdet = torch.sum(logs, [1, 2])

            return x, logdet

        else:
            x1 = (
                (x1 - m)
                * torch.exp(-logs)
                * x_mask
            )

            x = torch.cat([x0, x1], 1)

            return x, torch.zeros(
                (x.size(0),),
                device=x.device,
                dtype=x.dtype,
            )

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class Flip(nn.Module):
    """
    Inverte a ordem dos canais.

    Necessário para o normalizing flow funcionar corretamente.
    """

    def forward(
        self,
        x: torch.Tensor,
        *args,
        reverse: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.flip(x, [1])

        if not reverse:
            logdet = torch.zeros(
                x.size(0),
                device=x.device,
                dtype=x.dtype,
            )

            return x, logdet

        else:
            return x, torch.zeros(
                (1,),
                device=x.device,
                dtype=x.dtype,
            )


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()

        self.n_flows = n_flows
        self.flows = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )

            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                if isinstance(flow, ResidualCouplingLayer):
                    x, _ = flow(
                        x,
                        x_mask,
                        g=g,
                        reverse=False,
                    )
                else:
                    x, _ = flow(x, reverse=False)
        else:
            for flow in reversed(self.flows):
                if isinstance(flow, ResidualCouplingLayer):
                    x, _ = flow(
                        x,
                        x_mask,
                        g=g,
                        reverse=True,
                    )
                else:
                    x, _ = flow(x, reverse=True)

        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            if isinstance(
                self.flows[i * 2],
                ResidualCouplingLayer,
            ):
                self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()

        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )

        self.proj = nn.Conv1d(
            hidden_channels,
            out_channels * 2,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(
            commons_mask(x_lengths, x.size(2)),
            1,
        ).to(x.dtype)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)

        stats = self.proj(x) * x_mask

        m, logs = torch.split(
            stats,
            self.out_channels,
            dim=1,
        )

        z = (
            m + torch.randn_like(m) * torch.exp(logs)
        ) * x_mask

        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
# ============================================================================
# BLOCOS DO DISCRIMINADOR
# ============================================================================

class ResBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()

        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=(
                        kernel_size * dilation[0] - dilation[0]
                    ) // 2,
                )
            ),
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=(
                        kernel_size * dilation[1] - dilation[1]
                    ) // 2,
                )
            ),
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=(
                        kernel_size * dilation[2] - dilation[2]
                    ) // 2,
                )
            ),
        ])

        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                )
            ),
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                )
            ),
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                )
            ),
        ])

        self.lrelu_slope = 0.1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)

            if x_mask is not None:
                xt = xt * x_mask

            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)

            if x_mask is not None:
                xt = xt * x_mask

            xt = c2(xt)
            x = xt + x

        if x_mask is not None:
            x = x * x_mask

        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)

        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)


class SourceModuleHnNSF(nn.Module):
    """
    Gerador de harmônicos real com F0.

    harmonic_num=0 reproduce a arquitetura padrão do RVC para
    compatibilidade com conversores externos.
    """

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()

        self.sampling_rate = int(sampling_rate)
        self.upsample_scale = int(upsample_scale)
        self.harmonic_num = int(harmonic_num)
        self.sine_amp = float(sine_amp)
        self.noise_std = float(noise_std)
        self.voiced_threshold = float(voiced_threshold)

        self.l_linear = nn.Linear(
            harmonic_num + 1,
            1,
        )

        self.l_tanh = nn.Tanh()

    def _f0_to_uv(
        self,
        f0: torch.Tensor,
    ) -> torch.Tensor:
        return (f0 > self.voiced_threshold).float()

    def forward(
        self,
        f0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f0: (batch, T_frame) em frame-rate do encoder

        Returns:
            sine: (batch, 1, T_sample)
            uv: (batch, 1, T_sample)
            noise: (batch, 1, T_sample)
        """
        with torch.no_grad():
            f0_up = F.interpolate(
                f0.unsqueeze(1),
                scale_factor=float(self.upsample_scale),
                mode="nearest",
            ).squeeze(1)

            uv = self._f0_to_uv(f0_up).unsqueeze(-1)

            f0_buf = f0_up.unsqueeze(-1).repeat(
                1,
                1,
                self.harmonic_num + 1,
            )

            for idx in range(self.harmonic_num + 1):
                f0_buf[:, :, idx] = f0_up * (idx + 1)

            rad_values = (
                f0_buf / float(self.sampling_rate)
            ) % 1.0

            rand_ini = torch.rand(
                f0_buf.shape[0],
                f0_buf.shape[2],
                device=f0.device,
            )

            rand_ini[:, 0] = 0

            rad_values[:, 0, :] = (
                rad_values[:, 0, :] + rand_ini
            )

            phase = (
                torch.cumsum(rad_values, dim=1)
                * 2
                * math.pi
            )

            sines = torch.sin(phase) * self.sine_amp

            noise_amp = (
                uv * self.noise_std
                + (1 - uv) * (self.sine_amp / 3.0)
            )

            noise = noise_amp * torch.randn_like(sines)

            sine_waves = sines * uv + noise

        sine_merge = self.l_tanh(
            self.l_linear(sine_waves)
        )

        return (
            sine_merge.transpose(1, 2),
            uv.transpose(1, 2),
            noise.transpose(1, 2),
        )


class GeneratorNSF(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int,
        sr: int,
        is_half: bool = False,
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.Conv1d(
            initial_channel,
            upsample_initial_channel,
            7,
            1,
            padding=3,
        )

        resblock_class = ResBlock1

        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()

        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))

            for k, d in zip(
                resblock_kernel_sizes,
                resblock_dilation_sizes,
            ):
                self.resblocks.append(
                    resblock_class(ch, k, d)
                )

        self.conv_post = nn.Conv1d(
            ch,
            1,
            7,
            1,
            padding=3,
            bias=False,
        )

        self.ups.apply(
            lambda m: nn.init.normal_(
                m.weight,
                0.0,
                0.01,
            ) if hasattr(m, "weight") else None
        )

        if gin_channels != 0:
            self.cond = nn.Conv1d(
                gin_channels,
                upsample_initial_channel,
                1,
            )

        total_upsample = 1

        for u in upsample_rates:
            total_upsample *= u

        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr if isinstance(sr, int) else 40000,
            upsample_scale=total_upsample,
        )

        self.noise_convs = nn.ModuleList()

        for i, u in enumerate(upsample_rates):
            ch = upsample_initial_channel // (2 ** (i + 1))

            remaining = upsample_rates[i + 1:]
            stride = (
                int(np.prod(remaining))
                if len(remaining) > 0
                else 1
            )

            if stride > 1:
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        ch,
                        kernel_size=stride * 2,
                        stride=stride,
                        padding=stride // 2,
                    )
                )
            else:
                self.noise_convs.append(
                    nn.Conv1d(1, ch, kernel_size=1)
                )

    def forward(
        self,
        x: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv_pre(x)

        if g is not None and hasattr(self, "cond"):
            x = x + self.cond(g)

        har_source = None

        if f0 is not None:
            f0_in = (
                f0.squeeze(1)
                if f0.dim() == 3
                else f0
            )

            har_source, _, _ = self.m_source(f0_in)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            if har_source is not None:
                x_noise = self.noise_convs[i](
                    har_source
                )

                if x_noise.size(-1) != x.size(-1):
                    min_len = min(
                        x_noise.size(-1),
                        x.size(-1),
                    )

                    x = x[..., :min_len]
                    x_noise = x_noise[..., :min_len]

                x = x + x_noise

            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[
                        i * self.num_kernels + j
                    ](x)
                else:
                    xs += self.resblocks[
                        i * self.num_kernels + j
                    ](x)

            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)

        return torch.tanh(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            nn.utils.remove_weight_norm(layer)

        for layer in self.resblocks:
            layer.remove_weight_norm()


# ============================================================================
# MODELO SINTETIZADOR RVC COMPLETO
# ============================================================================

class SynthesizerTrnMs768NSFsid(nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        is_half: bool = False,
        phone_channels: int = 768,
        **kwargs,
    ):
        super().__init__()

        self.segment_size = segment_size

        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

        self.enc_p = TextEncoder768(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=is_half,
        )

        # O WN do flow usa n_layers=3, não 4.
        # Confirmado contra RVC-Project e Applio.
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        sid: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple,
    ]:
        """
        Args:
            phone: (batch, time, 768) - features do HuBERT
            phone_lengths: (batch,)
            pitch: (batch, time) - pitch coarse
            pitchf: (batch, time) - pitch fino
            spec: (batch, spec_dim, time)
            spec_lengths: (batch,)
            sid: (batch,)

        Returns:
            o: (batch, 1, time_sample)
            ids_slice: índices dos slices
            phone_mask: máscara de telefone
            spec_mask: máscara de espectro
            (z, z_p, m_p, logs_p, m_q, logs_q): dados do flow
        """
        g = self.emb_g(sid).unsqueeze(-1)

        m_p, logs_p, phone_mask = self.enc_p(
            phone,
            pitch,
            phone_lengths,
        )

        z, m_q, logs_q, spec_mask = self.enc_q(
            spec,
            spec_lengths,
            g=g,
        )

        # O flow opera no domínio espectral.
        # Use spec_mask, não phone_mask.
        z_p = self.flow(z, spec_mask, g=g)

        z_slice, ids_slice = commons_rand_slice(
            z,
            spec_lengths,
            self.segment_size,
        )

        pitchf_slice = commons_slice_segments(
            pitchf.unsqueeze(1),
            ids_slice,
            self.segment_size,
        )

        o = self.dec(
            z_slice,
            f0=pitchf_slice,
            g=g,
        )

        return (
            o,
            ids_slice,
            phone_mask,
            spec_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )

    def remove_weight_norm(self):
        self.enc_p.encoder.remove_weight_norm()
        self.enc_q.remove_weight_norm()
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()


# ============================================================================
# DISCRIMINADORES
# ============================================================================

class DiscriminatorS(nn.Module):
    def __init__(
        self,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        norm_f = (
            nn.utils.spectral_norm
            if use_spectral_norm
            else nn.utils.weight_norm
        )

        self.convs = nn.ModuleList([
            norm_f(
                nn.Conv1d(1, 16, 15, 1, padding=7)
            ),
            norm_f(
                nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)
            ),
            norm_f(
                nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)
            ),
            norm_f(
                nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)
            ),
            norm_f(
                nn.Conv1d(
                    1024,
                    1024,
                    41,
                    4,
                    groups=256,
                    padding=20,
                )
            ),
            norm_f(
                nn.Conv1d(1024, 1024, 5, 1, padding=2)
            ),
        ])

        self.conv_post = norm_f(
            nn.Conv1d(1024, 1, 3, 1, padding=1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        self.period = period

        norm_f = (
            nn.utils.spectral_norm
            if use_spectral_norm
            else nn.utils.weight_norm
        )

        self.convs = nn.ModuleList([
            norm_f(
                nn.Conv2d(
                    1,
                    32,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                )
            ),
            norm_f(
                nn.Conv2d(
                    32,
                    128,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                )
            ),
            norm_f(
                nn.Conv2d(
                    128,
                    512,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                )
            ),
            norm_f(
                nn.Conv2d(
                    512,
                    1024,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                )
            ),
            norm_f(
                nn.Conv2d(
                    1024,
                    1024,
                    (5, 1),
                    1,
                    padding=(2, 0),
                )
            ),
        ])

        self.conv_post = norm_f(
            nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        batch_size, channels, time = x.shape

        if time % self.period != 0:
            n_pad = self.period - (time % self.period)

            x = F.pad(x, (0, n_pad), "reflect")
            time = time + n_pad

        x = x.view(
            batch_size,
            channels,
            time // self.period,
            self.period,
        )

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(
        self,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        periods = [2, 3, 5, 7, 11, 17, 23, 37]

        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm)]
            + [
                DiscriminatorP(p, use_spectral_norm)
                for p in periods
            ]
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for discriminator in self.discriminators:
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# ============================================================================
# FUNÇÕES DE PERDA
# ============================================================================

def feature_loss(
    fmap_r: List[List[torch.Tensor]],
    fmap_g: List[List[torch.Tensor]],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=fmap_r[0][0].device)

    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(
                torch.abs(
                    rl.float().detach() - gl.float()
                )
            )

    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> torch.Tensor:
    loss = torch.tensor(0.0)

    for dr, dg in zip(
        disc_real_outputs,
        disc_generated_outputs,
    ):
        dr = dr.float()
        dg = dg.float()

        loss = loss + torch.mean((1 - dr) ** 2) + torch.mean(
            dg ** 2
        )

    return loss


def generator_loss(
    disc_outputs: List[torch.Tensor],
) -> torch.Tensor:
    loss = torch.tensor(0.0)

    for dg in disc_outputs:
        dg = dg.float()
        loss = loss + torch.mean((1 - dg) ** 2)

    return loss


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = (
        logs_p
        - logs_q
        - 0.5
        + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    )

    return torch.sum(kl * z_mask) / torch.sum(z_mask)


# ============================================================================
# ESPECTROGRAMA
# ============================================================================

def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    if y.dim() == 1:
        y = y.unsqueeze(0)

    y = y.squeeze(0)

    pad = (n_fft - hop_size) // 2

    y = F.pad(
        y.unsqueeze(0),
        (pad, pad),
        mode="reflect",
    ).squeeze(0)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=torch.hann_window(win_size).to(y.device),
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    return torch.sqrt(
        spec.real.pow(2) + spec.imag.pow(2) + 1e-7
    )


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: int,
    fmax: int,
) -> torch.Tensor:
    try:
        from librosa.filters import mel as librosa_mel_fn

        mel = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )

        mel_basis = torch.from_numpy(mel).to(
            dtype=spec.dtype,
            device=spec.device,
        )

    except Exception:
        mel_basis = torch.randn(
            num_mels,
            n_fft // 2 + 1,
        ).to(
            dtype=spec.dtype,
            device=spec.device,
        ) * 0.01

    return torch.matmul(mel_basis, spec).clamp_min(
        1e-5
    ).log()


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int,
    center: bool = False,
) -> torch.Tensor:
    spec = spectrogram_torch(
        y,
        n_fft,
        sampling_rate,
        hop_size,
        win_size,
        center,
    )

    return spec_to_mel_torch(
        spec,
        n_fft,
        num_mels,
        sampling_rate,
        fmin,
        fmax,
    )


# ============================================================================
# DATASET
# ============================================================================

def load_wav_to_torch(
    full_path: str,
) -> Tuple[torch.Tensor, int]:
    sampling_rate, data = wavfile.read(full_path)

    data = data.astype(np.float32)

    if np.abs(data).max() > 2.5:
        data = data / 32768.0

    return torch.FloatTensor(data), sampling_rate


class TextAudioLoaderMultiNSFsid(
    torch.utils.data.Dataset
):
    def __init__(
        self,
        audiopaths_and_text: str,
        hparams,
    ):
        self.audiopaths_and_text = []

        with open(audiopaths_and_text, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split("|")

                if len(parts) >= 5:
                    self.audiopaths_and_text.append(parts)

        self.sampling_rate = (
            hparams.data.sampling_rate
        )

        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length

        self.lengths = [
            os.path.getsize(p[0])
            // (3 * self.hop_length)
            for p in self.audiopaths_and_text
        ]

    def get_audio_text_pair(
        self,
        audiopath_and_text,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        file = audiopath_and_text[0]
        phone_path = audiopath_and_text[1]
        pitch_path = audiopath_and_text[2]
        pitchf_path = audiopath_and_text[3]
        dv = audiopath_and_text[4]

        phone = torch.FloatTensor(
            np.load(phone_path)
        )

        pitch = torch.LongTensor(
            np.load(pitch_path)
        )

        pitchf = torch.FloatTensor(
            np.load(pitchf_path)
        )

        spec, wav = self.get_audio(file)

        dv = torch.LongTensor([int(dv)])

        len_phone = phone.size(0)
        len_spec = spec.size(-1)

        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length

            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return spec, wav, phone, pitch, pitchf, dv

    def get_audio(
        self,
        filename: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, sampling_rate = load_wav_to_torch(filename)

        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Sample rate mismatch: "
                f"{sampling_rate} vs {self.sampling_rate}"
            )

        audio_norm = audio.unsqueeze(0)

        spec_filename = filename.replace(
            ".wav",
            ".spec.pt",
        )

        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename)

            except Exception:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )

                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename)

        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )

            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        return spec, audio_norm

    def __getitem__(
        self,
        index: int,
    ):
        return self.get_audio_text_pair(
            self.audiopaths_and_text[index]
        )

    def __len__(self) -> int:
        return len(self.audiopaths_and_text)


class TextAudioCollateMultiNSFsid:
    def __call__(
        self,
        batch,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _, ids_sorted = torch.sort(
            torch.LongTensor(
                [x[0].size(1) for x in batch]
            ),
            dim=0,
            descending=True,
        )

        max_spec_len = max(
            [x[0].size(1) for x in batch]
        )

        max_wave_len = max(
            [x[1].size(1) for x in batch]
        )

        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(
            len(batch),
            batch[0][0].size(0),
            max_spec_len,
        ).zero_()

        wave_padded = torch.FloatTensor(
            len(batch),
            1,
            max_wave_len,
        ).zero_()

        max_phone_len = max(
            [x[2].size(0) for x in batch]
        )

        phone_lengths = torch.LongTensor(len(batch))

        phone_padded = torch.FloatTensor(
            len(batch),
            max_phone_len,
            batch[0][2].shape[1],
        ).zero_()

        pitch_padded = torch.LongTensor(
            len(batch),
            max_phone_len,
        ).zero_()

        pitchf_padded = torch.FloatTensor(
            len(batch),
            max_phone_len,
        ).zero_()

        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted)):
            row = batch[ids_sorted[i]]

            spec_padded[i, :, :row[0].size(1)] = row[0]
            spec_lengths[i] = row[0].size(1)

            wave_padded[i, :, :row[1].size(1)] = row[1]
            wave_lengths[i] = row[1].size(1)

            phone_padded[i, :row[2].size(0), :] = row[2]
            phone_lengths[i] = row[2].size(0)

            pitch_padded[i, :row[3].size(0)] = row[3]
            pitchf_padded[i, :row[4].size(0)] = row[4]

            sid[i] = row[5]

        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


# ============================================================================
# CHECKPOINTS
# ============================================================================

# ====== CHECKPOINTS ========
def save_resume_checkpoint(
    path: Path,
    net_g: nn.Module,
    net_d: nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    scheduler_g: Optional[torch.optim.lr_scheduler.LRScheduler],
    scheduler_d: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler,
    epoch: int,
) -> None:
    """Salva checkpoint completo de retomada."""
    state = {
        "epoch": int(epoch),
        "net_g": (
            net_g.module.state_dict()
            if hasattr(net_g, "module")
            else net_g.state_dict()
        ),
        "net_d": (
            net_d.module.state_dict()
            if hasattr(net_d, "module")
            else net_d.state_dict()
        ),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "scheduler_g": (
            scheduler_g.state_dict()
            if scheduler_g is not None
            else None
        ),
        "scheduler_d": (
            scheduler_d.state_dict()
            if scheduler_d is not None
            else None
        ),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)

def load_resume_checkpoint(
    path: Path,
    net_g: nn.Module,
    net_d: nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    scheduler_g: Optional[torch.optim.lr_scheduler.LRScheduler],
    scheduler_d: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler,
) -> int:
    """Carrega checkpoint de retomada. Retorna próximo epoch."""
    path = Path(path)

    if not path.exists():
        return 1

    checkpoint = torch.load(path, map_location="cpu")

    target_g = (
        net_g.module
        if hasattr(net_g, "module")
        else net_g
    )

    target_d = (
        net_d.module
        if hasattr(net_d, "module")
        else net_d
    )

    target_g.load_state_dict(
        checkpoint["net_g"],
        strict=False,
    )

    target_d.load_state_dict(
        checkpoint["net_d"],
        strict=False,
    )

    optim_g.load_state_dict(checkpoint["optim_g"])
    optim_d.load_state_dict(checkpoint["optim_d"])

    if scheduler_g is not None and checkpoint.get("scheduler_g") is not None:
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])

    if scheduler_d is not None and checkpoint.get("scheduler_d") is not None:
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])

    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])

    return int(checkpoint["epoch"]) + 1


def savee(
    ckpt: dict,
    sr: str,
    if_f0: int,
    name: str,
    epoch: int,
    version: str,
    hps,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Salva o modelo final em formato RVC.

    Retorna o caminho do arquivo ou uma string de erro.
    """
    try:
        output_dir = Path(output_dir or WEIGHTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        opt = {
            "weight": {},
            "config": [
                hps.data.filter_length // 2 + 1,
                32,
                hps.model.inter_channels,
                hps.model.hidden_channels,
                hps.model.filter_channels,
                hps.model.n_heads,
                hps.model.n_layers,
                hps.model.kernel_size,
                hps.model.p_dropout,
                hps.model.resblock,
                hps.model.resblock_kernel_sizes,
                hps.model.resblock_dilation_sizes,
                hps.model.upsample_rates,
                hps.model.upsample_initial_channel,
                hps.model.upsample_kernel_sizes,
                hps.model.spk_embed_dim,
                hps.model.gin_channels,
                hps.data.sampling_rate,
            ],
            "info": f"{epoch}epoch",
            "sr": sr,
            "f0": if_f0,
            "version": version,
        }

        for key, value in ckpt.items():
            if "enc_q" not in key:
                opt["weight"][key] = value.half()

        output_path = output_dir / f"{name}.pth"
        temporary = output_path.with_suffix(".tmp")

        torch.save(opt, temporary)
        os.replace(temporary, output_path)

        if (
            not output_path.exists()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Arquivo de saída vazio."
            )

        return str(output_path)

    except Exception as exc:
        logger.exception("Falha ao salvar checkpoint final")
        return f"ERRO: {exc}"


# ============================================================================
# TREINAMENTO
# ============================================================================

def train_and_evaluate(
    rank: int,
    epoch: int,
    hps,
    net_g: nn.Module,
    net_d: nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    train_loader,
    scaler,
    writer,
    job_id: Optional[str] = None,
) -> None:
    from torch.cuda.amp import autocast

    net_g.train()
    net_d.train()

    global_step = (epoch - 1) * len(train_loader)

    for batch_idx, (
        phone,
        phone_lengths,
        pitch,
        pitchf,
        spec,
        spec_lengths,
        wave,
        wave_lengths,
        sid,
    ) in enumerate(train_loader):
        if torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(
                rank,
                non_blocking=True,
            )

            pitch = pitch.cuda(rank, non_blocking=True)
            pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)

            spec_lengths = spec_lengths.cuda(
                rank,
                non_blocking=True,
            )

            wave = wave.cuda(rank, non_blocking=True)

        is_half = (
            hps.train.fp16_run
            and torch.cuda.is_available()
        )

        with autocast(enabled=is_half):
            (
                y_hat,
                ids_slice,
                phone_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                sid,
            )

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y_mel = commons_slice_segments(
                mel,
                ids_slice,
                hps.train.segment_size // hps.data.hop_length,
            )

            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

            if is_half:
                y_hat_mel = y_hat_mel.half()

            wave_slice = commons_slice_segments(
                wave,
                ids_slice * hps.data.hop_length,
                hps.train.segment_size,
            )

            y_d_hat_r, y_d_hat_g, _, _ = net_d(
                wave_slice,
                y_hat.detach(),
            )

            with autocast(enabled=False):
                loss_disc = discriminator_loss(
                    y_d_hat_r,
                    y_d_hat_g,
                )

        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)

        torch.nn.utils.clip_grad_norm_(
            net_d.parameters(),
            float("inf"),
        )

        scaler.step(optim_d)

        with autocast(enabled=is_half):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = (
                net_d(wave_slice, y_hat)
            )

            with autocast(enabled=False):
                loss_mel = (
                    F.l1_loss(y_mel, y_hat_mel)
                    * hps.train.c_mel
                )

                loss_kl = (
                    kl_loss(
                        z_p,
                        logs_q,
                        m_p,
                        logs_p,
                        z_mask,
                    )
                    * hps.train.c_kl
                )

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen = generator_loss(y_d_hat_g)
                loss_gen_all = (
                    loss_gen + loss_fm + loss_mel + loss_kl
                )

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)

        torch.nn.utils.clip_grad_norm_(
            net_g.parameters(),
            float("inf"),
        )

        scaler.step(optim_g)
        scaler.update()

        if rank == 0 and (
            global_step % hps.train.log_interval == 0
        ):
            lr = optim_g.param_groups[0]["lr"]

            msg = (
                f"Epoch {epoch} "
                f"[{100.0 * batch_idx / len(train_loader):.0f}%] | "
                f"loss_mel={loss_mel.item():.3f} "
                f"loss_kl={loss_kl.item():.3f} "
                f"lr={lr:.6f}"
            )

            logger.info(msg)

            if job_id:
                append_log(job_id, msg)

            if writer:
                writer.add_scalar(
                    "loss/g/total",
                    loss_gen_all,
                    global_step,
                )

                writer.add_scalar(
                    "loss/d/total",
                    loss_disc,
                    global_step,
                )

                writer.add_scalar(
                    "learning_rate",
                    lr,
                    global_step,
                )

            progress = min(
                0.5 + 0.5 * (epoch / hps.train.epochs),
                0.99,
            )

        if job_id:
            update_job(
                job_id,
                status=STATUS_TRAINING,
                progress=min(0.99, (epoch / max(1, hps.train.epochs))),
                message=f"Epoch {epoch}/{hps.train.epochs}",
                current_epoch=epoch,
        )

        global_step += 1

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        model_dir = Path(hps.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        save_resume_checkpoint(
            model_dir / "resume.pt",
            net_g,
            net_d,
            optim_g,
            optim_d,
            None,
            None,
            scaler,
            epoch,
        )

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()

        savee_result = savee(
            ckpt,
            hps.sample_rate,
            1,
            f"{hps.name}_e{epoch}_s{global_step}",
            epoch,
            hps.version,
            hps,
            output_dir=WEIGHTS_DIR / job_id
            if job_id
            else None,
        )

        if not isinstance(savee_result, str) or savee_result.startswith(
            "ERRO:"
        ):
            msg = (
                f"AVISO: Falha ao salvar "
                f"checkpoint intermediário: {savee_result}"
            )

            logger.warning(msg)

            if job_id:
                append_log(job_id, msg)

    if rank == 0:
        logger.info(f"Epoch {epoch} concluído.")
        # ============================================================================
# HPARAMS
# ============================================================================

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)

            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__


def get_hparams_from_dict(config_dict: dict) -> HParams:
    return HParams(**config_dict)


def get_logger(
    model_dir: str,
    filename: str = "train.log",
):
    logger = logging.getLogger(
        os.path.basename(model_dir)
    )

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s"
    )

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Evita handlers duplicados
    if not logger.handlers:
        handler = logging.FileHandler(
            model_dir / filename,
            encoding="utf8",
        )

        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# ============================================================================
# ZIP UTILS
# ============================================================================

def make_zip(
    model_dir: str,
    job_id: str,
    name: str,
    metadata: dict,
) -> Tuple[str, bool]:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOADS_DIR / f"{job_id}_{name}.zip"
    weights_dir = WEIGHTS_DIR / job_id

    pth_files = sorted(weights_dir.glob("*.pth")) if weights_dir.exists() else []

    if not pth_files:
        raise RuntimeError("Nenhum modelo (.pth) foi salvo.")

    is_inference_ready = True
    index_files = sorted(Path(model_dir).glob("*.index"))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        best_pth = pth_files[-1]
        shutil.copy2(best_pth, temp_path / f"{name}.pth")

        if index_files:
            shutil.copy2(index_files[-1], temp_path / f"{name}.index")
            is_inference_ready = True
        else:
            logger.warning("Nenhum .index encontrado; ZIP do modelo será apenas .pth.")
            is_inference_ready = False

        config_file = Path(model_dir) / "config.json"
        if config_file.exists():
            shutil.copy2(config_file, temp_path / "config.json")

        metadata = dict(metadata)
        metadata["job_id"] = job_id
        metadata["inference_ready"] = is_inference_ready

        with open(temp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        archive_base = str(zip_path.with_suffix(""))
        shutil.make_archive(archive_base, "zip", temp_dir)

    return str(zip_path), is_inference_ready
        
# ============================================================================
# EXECUTOR DE TREINAMENTO
# ============================================================================
_TRAINING_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_TRAINING,
    thread_name_prefix="rvc-training",
)
_TRAINING_FUTURES: Dict[str, object] = {}
_TRAINING_FUTURES_LOCK = threading.RLock()


def _training_worker(
    job_id: str,
    model_name: str,
    author: str,
    description: str,
    tags: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sample_rate: str,
    f0_method: str,
    checkpoint_interval: int,
    audio_path: str,
    photo_path: Optional[str] = None,
) -> None:
    try:
        _run_training_pipeline(
            job_id=job_id,
            model_name=model_name,
            author=author,
            description=description,
            tags=tags,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            sample_rate=sample_rate,
            f0_method=f0_method,
            checkpoint_interval=checkpoint_interval,
            audio_path=audio_path,
            photo_path=photo_path,
        )
    except Exception as exc:
        logger.exception("Falha no job %s", job_id)
        append_log(job_id, f"ERRO: {exc}")
        update_job(
            job_id,
            status=STATUS_FAILED,
            message=f"Falha no treinamento: {exc}",
        )
    finally:
        flush_all_logs()

def start_training_job(
    job_id: str,
    model_name: str,
    author: str,
    description: str,
    tags: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sample_rate: str,
    f0_method: str,
    checkpoint_interval: int,
    audio_path: str,
    photo_path: Optional[str] = None,
) -> str:
    job = {
        "job_id": job_id,
        "model_name": model_name,
        "author": author,
        "description": description,
        "tags": tags,
        "status": STATUS_WAITING,
        "progress": 0.0,
        "message": "Job iniciado. Aguardando fila...",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "sample_rate": sample_rate,
        "f0_method": f0_method,
        "checkpoint_interval": int(checkpoint_interval),
        "audio_path": str(audio_path),
        "photo_path": str(photo_path) if photo_path else None,
        "created_at": datetime.now().isoformat(),
        "current_epoch": 0,
    }

    set_job(job_id, job)

    future = _TRAINING_EXECUTOR.submit(
        _training_worker,
        job_id,
        model_name,
        author,
        description,
        tags,
        epochs,
        batch_size,
        learning_rate,
        sample_rate,
        f0_method,
        checkpoint_interval,
        audio_path,
        photo_path,
    )

    with _TRAINING_FUTURES_LOCK:
        _TRAINING_FUTURES[job_id] = future
    
# ============================================================================
# PIPELINE PRINCIPAL DE TREINAMENTO
# ============================================================================

def _run_training_pipeline(
    job_id: str,
    model_name: str,
    author: str,
    description: str,
    tags: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sample_rate: str,
    f0_method: str,
    checkpoint_interval: int,
    audio_path: str,
    photo_path: Optional[str] = None,
) -> None:
    """
    Pipeline isolado por job_id.

    Cada job tem seu próprio diretório, pesos e checkpoints.
    """
    # Validações
    epochs = max(MIN_EPOCHS, min(epochs, MAX_EPOCHS))
    batch_size = max(1, min(batch_size, 128))
    checkpoint_interval = max(
        1,
        min(checkpoint_interval, epochs),
    )

    # Sanitize model name
    model_name_sanitized = sanitize_filename(model_name)

    # Diretório único por job
    model_name_dir = f"{model_name_sanitized}_{job_id}"
    exp_dir = str(MODELS_DIR / model_name_dir)

    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # Verificar resumção
    resume_path = (
        CHECKPOINTS_DIR / job_id / "resume.pt"
    )

    # Copiar áudio para exp_dir
    input_audio_dir = Path(exp_dir) / "input_audio"
    input_audio_dir.mkdir(parents=True, exist_ok=True)

    append_log(
        job_id,
        f"Copiando áudio para {input_audio_dir}...",
    )

    shutil.copy2(
        audio_path,
        input_audio_dir / "input.wav",
    )

    audio_path_for_pipeline = str(
        input_audio_dir / "input.wav"
    )

    # ====== STAGE 1: AUDIO PIPELINE ======
    update_job(
        job_id,
        status=STATUS_PREPROCESSING,
        progress=0.05,
        message="Convertendo e processando áudio...",
    )

    try:
        audio_result = process_audio_pipeline(
            audio_path_for_pipeline,
            exp_dir,
            SR_TARGET,
        )

        append_log(
            job_id,
            f"Áudio processado: "
            f"{audio_result['total_segments']} segmentos, "
            f"{format_duration(audio_result['total_duration'])}.",
        )

    except Exception as exc:
        append_log(job_id, f"ERRO no processamento de áudio: {exc}")
        raise RuntimeError(
            f"Falha ao processar áudio: {exc}"
        ) from exc

    # ====== STAGE 2: PRÉ-PROCESSAMENTO ======
    update_job(
        job_id,
        status=STATUS_PREPROCESSING,
        progress=0.1,
        message="Pré-processando...",
    )

    try:
        preprocess_result = preprocess_trainset(
            exp_dir,
            SR_TARGET,
            exp_dir,
            per=3.7,
        )

        append_log(
            job_id,
            f"Pré-processamento concluído: "
            f"{preprocess_result['written_segments']} segmentos salvos.",
        )

    except Exception as exc:
        append_log(job_id, f"ERRO no pré-processamento: {exc}")
        raise RuntimeError(
            f"Falha no pré-processamento: {exc}"
        ) from exc

    # ====== STAGE 3: F0 EXTRACTION ======
    update_job(
        job_id,
        status=STATUS_EXTRACTING_F0,
        progress=0.2,
        message="Extraindo F0...",
    )

    try:
        n_f0 = run_f0_extraction(exp_dir)
        append_log(job_id, f"F0 extraído de {n_f0} arquivos.")

    except Exception as exc:
        append_log(job_id, f"ERRO na extração de F0: {exc}")
        raise RuntimeError(
            f"Falha na extração de F0: {exc}"
        ) from exc

    # ====== STAGE 4: FEATURE EXTRACTION ======
    update_job(
        job_id,
        status=STATUS_EXTRACTING_FEATURES,
        progress=0.3,
        message="Extraindo features com HuBERT...",
    )

    try:
        n_features, used_real_hubert = (
            run_feature_extraction(
                exp_dir,
                version="v2",
                job_id=job_id,
            )
        )

        if not used_real_hubert:
            append_log(
                job_id,
                "AVISO: HuBERT não foi carregado. "
                "Features são projeção aleatória.",
            )

        append_log(
            job_id,
            f"Features extraídas de {n_features} arquivos.",
        )

    except Exception as exc:
        append_log(job_id, f"ERRO na extração de features: {exc}")
        raise RuntimeError(
            f"Falha na extração de features: {exc}"
        ) from exc

    # ====== STAGE 5: FAISS INDEX ======
    update_job(
        job_id,
        status=STATUS_TRAINING_INDEX,
        progress=0.4,
        message="Treinando índice FAISS...",
    )

    try:
        index_path = train_index(
            model_name_dir,
            version="v2",
            exp_dir_base=str(MODELS_DIR),
        )

        if index_path:
            append_log(
                job_id,
                f"Índice FAISS criado: {index_path}",
            )
        else:
            append_log(job_id, "Índice FAISS não foi criado.")

    except Exception as exc:
        append_log(
            job_id,
            f"AVISO: Falha ao criar índice FAISS: {exc}",
        )

    # ====== STAGE 6: FILELIST ======
    update_job(
        job_id,
        status=STATUS_TRAINING,
        progress=0.45,
        message="Gerando filelist...",
    )

    gt_wavs_dir = Path(exp_dir) / "0_gt_wavs"
    filelist_path = Path(exp_dir) / "filelist.txt"

    filelist_lines = []

    for wav_file in sorted(gt_wavs_dir.glob("*.wav")):
        base_name = wav_file.stem

        phone_file = Path(exp_dir) / "3_feature768" / (
            f"{base_name}.npy"
        )

        pitch_file = Path(exp_dir) / "2a_f0" / (
            f"{base_name}.npy"
        )

        pitchf_file = Path(exp_dir) / "2b-f0nsf" / (
            f"{base_name}.npy"
        )

        if all([
            phone_file.exists(),
            pitch_file.exists(),
            pitchf_file.exists(),
        ]):
            filelist_lines.append(
                f"{wav_file}|{phone_file}|"
                f"{pitch_file}|{pitchf_file}|0"
            )

    if not filelist_lines:
        raise RuntimeError(
            "Nenhum arquivo válido para treinamento."
        )

    with open(filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(filelist_lines))

    append_log(
        job_id,
        f"Filelist criado com {len(filelist_lines)} amostras.",
    )

    # ====== STAGE 7: CONFIG ======
    config = {
        "train": {
            "learning_rate": learning_rate,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": batch_size,
            "num_workers": 2,
            "segment_size": 12800,
            "init_lr_ratio": 1.0,
            "warmup_epochs": 0,
            "epochs": epochs,
            "log_interval": 10,
            "eval_interval": 100,
            "save_interval": max(1, epochs // 10),
            "decay_epoch": 0,
            "fp16_run": (
                torch.cuda.is_available()
            ),
            "lr_decay": 0.999875,
            "seed": 1234,
            "c_mel": 45.0,
            "c_kl": 1.0,
        },
        "data": {
            "n_fft": 2048,
            "n_mel_channels": 80,
            "sampling_rate": SR_TARGET,
            "hop_length": 512,
            "win_length": 2048,
            "f_pmax": 12000.0,
            "f_pmin": 40.0,
            "mel_fmin": 0,
            "mel_fmax": 12000,
            "filter_length": 2048,
            "training_files": str(filelist_path),
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "spk_embed_dim": 256,
            "gin_channels": 256,
        },
    }

    hps = get_hparams_from_dict(config)
    hps.model_dir = exp_dir
    hps.name = model_name_sanitized
    hps.sample_rate = sample_rate
    hps.version = "v2"
    hps.sr_numeric = SR_TARGET
    hps.save_every_epoch = checkpoint_interval

    config_path = Path(exp_dir) / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    append_log(job_id, "Config salvo.")

    # ====== STAGE 8: TRAINING ======
    update_job(
        job_id,
        status=STATUS_TRAINING,
        progress=0.5,
        message="Iniciando treinamento GAN...",
    )

    try:
        run_training(hps, job_id)
    except Exception as exc:
        append_log(job_id, f"ERRO no treinamento: {exc}")
        raise RuntimeError(f"Falha no treinamento: {exc}") from exc

    update_job(
        job_id,
        status=STATUS_DONE,
        progress=1.0,
        message="Treinamento concluído com sucesso!",
        current_epoch=epochs,
    )

    # ====== STAGE 9: ZIP ======
    update_job(
        job_id,
        status=STATUS_DONE,
        progress=0.95,
        message="Gerando download...",
    )

    try:
        metadata = {
            "job_id": job_id,
            "model_name": model_name,
            "author": author,
            "description": description,
            "tags": tags,
            "epochs": epochs,
            "sample_rate": sample_rate,
            "f0_method": f0_method,
            "created_at": datetime.now().isoformat(),
        }

        zip_path, is_ready = make_zip(
            exp_dir,
            job_id,
            model_name_sanitized,
            metadata,
        )

        update_job(
            job_id,
            download_path=zip_path,
            is_inference_ready=is_ready,
        )

        append_log(
            job_id,
            f"Download pronto: {zip_path}",
        )

    except Exception as exc:
        append_log(job_id, f"AVISO: Falha ao gerar ZIP: {exc}")


def run_training(hps, job_id: Optional[str] = None) -> None:
    logger_inst = get_logger(hps.model_dir)
    logger_inst.info(f"Device: {DEVICE_LABEL}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=hps.model_dir)
    except Exception:
        writer = None

    training_is_half = hps.train.fp16_run and torch.cuda.is_available()

    dataset = TextAudioLoaderMultiNSFsid(
        hps.data.training_files,
        hps,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset vazio.")

    train_loader = DataLoader(
        dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TextAudioCollateMultiNSFsid(),
        drop_last=True,
    )

    net_g = SynthesizerTrnMs768NSFsid(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
        is_half=training_is_half,
        sr=getattr(hps, "sr_numeric", 40000),
    ).to(device)

    net_d = MultiPeriodDiscriminator().to(device)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=hps.train.lr_decay,
    )

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=hps.train.lr_decay,
    )

    from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=training_is_half)

    resume_path = CHECKPOINTS_DIR / job_id / "resume.pt" if job_id else None
    epoch_str = 1

    if resume_path and resume_path.exists():
        try:
            epoch_str = load_resume_checkpoint(
                resume_path,
                net_g,
                net_d,
                optim_g,
                optim_d,
                scheduler_g,
                scheduler_d,
                scaler,
            )
            logger_inst.info(f"Retomando do epoch {epoch_str}")
        except Exception as exc:
            logger_inst.warning(f"Não foi possível retomar: {exc}")
            epoch_str = 1

    for epoch in range(epoch_str, hps.train.epochs + 1):
        net_g.train()
        net_d.train()

        if job_id:
            update_job(
                job_id,
                status=STATUS_TRAINING,
                progress=min(0.99, (epoch / max(1, hps.train.epochs))),
                message=f"Epoch {epoch}/{hps.train.epochs}",
                current_epoch=epoch,
            )

        for batch_idx, (
            phone,
            phone_lengths,
            pitch,
            pitchf,
            spec,
            spec_lengths,
            wave,
            wave_lengths,
            sid,
        ) in enumerate(train_loader):

            phone = phone.to(device)
            phone_lengths = phone_lengths.to(device)
            pitch = pitch.to(device)
            pitchf = pitchf.to(device)
            spec = spec.to(device)
            spec_lengths = spec_lengths.to(device)
            wave = wave.to(device)
            sid = sid.to(device)

            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=training_is_half,
            ):
                y_hat, ids_slice, phone_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    sid,
                )

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y_mel = commons_slice_segments(
                    mel,
                    ids_slice,
                    hps.train.segment_size // hps.data.hop_length,
                )

                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                wave_slice = commons_slice_segments(
                    wave,
                    ids_slice * hps.data.hop_length,
                    hps.train.segment_size,
                )

                if y_hat.size(-1) != wave_slice.size(-1):
                    min_len = min(y_hat.size(-1), wave_slice.size(-1))
                    if min_len > 0:
                        y_hat = y_hat[..., :min_len]
                        wave_slice = wave_slice[..., :min_len]

                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave_slice, y_hat.detach())
                loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)

            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            torch.nn.utils.clip_grad_norm_(net_d.parameters(), float("inf"))
            scaler.step(optim_d)

            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=training_is_half,
            ):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave_slice, y_hat)

                target_len = min(y_mel.size(-1), y_hat_mel.size(-1))
                if target_len > 0:
                    y_mel = y_mel[..., :target_len]
                    y_hat_mel = y_hat_mel[..., :target_len]

                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), float("inf"))
            scaler.step(optim_g)
            scaler.update()

            if writer is not None:
                writer.add_scalar(
                    "loss/g/total",
                    loss_gen_all.item(),
                    epoch * len(train_loader) + batch_idx,
                )
                writer.add_scalar(
                    "loss/d/total",
                    loss_disc.item(),
                    epoch * len(train_loader) + batch_idx,
                )

        scheduler_g.step()
        scheduler_d.step()

        if resume_path and epoch % hps.save_every_epoch == 0:
            save_resume_checkpoint(
                resume_path,
                net_g,
                net_d,
                optim_g,
                optim_d,
                scheduler_g,
                scheduler_d,
                scaler,
                epoch,
            )

        if epoch >= hps.train.epochs:
            logger_inst.info("Treinamento concluído!")

            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            final_weights_dir = WEIGHTS_DIR / job_id if job_id else WEIGHTS_DIR

            savee_result = savee(
                ckpt,
                hps.sample_rate,
                1,
                hps.name,
                epoch,
                hps.version,
                hps,
                output_dir=final_weights_dir,
            )

            if isinstance(savee_result, str) and savee_result.startswith("ERRO:"):
                raise RuntimeError(f"Falha ao salvar modelo final: {savee_result}")

            if job_id:
                append_log(job_id, f"Modelo final salvo em {savee_result}")
                update_job(
                    job_id,
                    status=STATUS_DONE,
                    progress=1.0,
                    message="Treinamento concluído com sucesso!",
                    current_epoch=epoch,
                    epochs=hps.train.epochs,
                )

            break

    if writer:
        writer.close()
        

# ============================================================================ 
# APLICAÇÃO WEB (SEM GRADIO, SEM ABA TRAVADA)
# ============================================================================

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="RVC Training")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PAGE = r'''<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RVC Training</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    .tabs { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
    .tab-btn {
      border: 1px solid #334155;
      background: #111827;
      color: #e2e8f0;
      padding: 10px 18px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 700;
      min-width: 130px;
    }
    .tab-btn.active {
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      border: none;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .grid { display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 20px; }
    .card {
      background: #111827;
      border: 1px solid #334155;
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 8px 18px rgba(0,0,0,.2);
    }
    input, select, textarea, button {
      width: 100%;
      box-sizing: border-box;
      margin-top: 8px;
      margin-bottom: 10px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #475569;
      background: #0b1220;
      color: #e2e8f0;
    }
    button {
      cursor: pointer;
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      border: none;
      font-weight: 700;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0,1fr));
      gap: 12px;
    }
    .status {
      white-space: pre-wrap;
      line-height: 1.5;
      font-family: ui-monospace, SFMono-Regular, monospace;
      background: #020817;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 12px;
      min-height: 120px;
      overflow-y: auto;
      max-height: 300px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }
    th, td {
      border-bottom: 1px solid #334155;
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }
    .small { font-size: 0.85rem; opacity: 0.9; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎤 RVC Training</h1>

    <div class="tabs">
      <button type="button" class="tab-btn active" data-tab="training">Treinamento</button>
      <button type="button" class="tab-btn" data-tab="jobs">Jobs</button>
      <button type="button" class="tab-btn" data-tab="downloads">Downloads</button>
    </div>

    <div id="tab-training" class="tab-panel active">
      <div class="grid">
        <div class="card">
          <h3>📝 Treinamento</h3>

          <label>Nome do Modelo
            <input id="model_name" value="VoiceA" />
          </label>

          <label>Autor (opcional)
            <input id="author" value="Desconhecido" />
          </label>

          <label>Descrição (opcional)
            <textarea id="description" rows="3"></textarea>
          </label>

          <label>Tags (opcional)
            <input id="tags" value="rvc, treino" />
          </label>

          <div class="row">
            <label>Épocas
              <input id="epochs" type="number" min="50" max="1000" value="100" />
            </label>
            <label>Batch Size
              <input id="batch_size" type="number" min="1" max="128" value="16" />
            </label>
          </div>

          <div class="row">
            <label>Learning Rate
              <input id="learning_rate" type="number" step="0.0001" value="0.0002" />
            </label>
            <label>Checkpoint
              <input id="checkpoint_interval" type="number" min="1" max="50" value="10" />
            </label>
          </div>

          <label>Sample Rate
            <select id="sample_rate">
              <option value="40k" selected>40k</option>
              <option value="32k">32k</option>
              <option value="48k">48k</option>
            </select>
          </label>

          <label>F0
            <select id="f0_method">
              <option value="yin" selected>yin</option>
              <option value="dio">dio</option>
            </select>
          </label>

          <label>Arquivo de áudio/vídeo
            <input id="media_file" type="file" />
          </label>

          <div style="display:flex;gap:8px;align-items:center;">
            <button type="button" id="trainBtn">🚀 Iniciar Treinamento</button>
            <button type="button" id="reloadJobsBtnTraining" style="width:180px;background:linear-gradient(135deg,#10b981,#059669);">🔁 Recarregar Jobs</button>
          </div>

          <div id="trainStatus" class="status">Aguardando...</div>
        </div>
      </div>
    </div>

    <div id="tab-jobs" class="tab-panel">
      <div class="card">
        <h3>📊 Jobs (Auto-atualiza a cada 3s)</h3>
        <table>
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Modelo</th>
              <th>Status</th>
              <th>Progresso</th>
              <th>Época</th>
            </tr>
          </thead>
          <tbody id="jobsTable"></tbody>
        </table>

        <h3 style="margin-top: 18px;">📋 Logs do Job Selecionado</h3>
        <label>Job ID
          <input id="selectedJobId" placeholder="Clique em um job acima" readonly />
        </label>
        <button type="button" id="reloadLogsBtn">🔄 Recarregar Logs</button>
        <div id="logsView" class="status">Selecione um job...</div>
      </div>
    </div>

    <div id="tab-downloads" class="tab-panel">
      <div class="card">
        <h3>⬇️ Downloads dos modelos treinados</h3>

        <table>
          <thead>
            <tr>
              <th>Job</th>
              <th>Modelo</th>
              <th>Status</th>
              <th>Download</th>
            </tr>
          </thead>
          <tbody id="downloadsTable"></tbody>
        </table>

        <div id="downloadStatus" class="status" style="margin-top:18px;">Nenhum download disponível.</div>
      </div>
    </div>
  </div>

  <script>
    let selectedJobId = null;

    function showTab(tabName) {
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
      });
      document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === `tab-${tabName}`);
      });
    }

    async function fetchJson(url, options = {}) {
      const res = await fetch(url, options);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Erro ${res.status}`);
      }
      return res.json();
    }

    async function loadJobs() {
      try {
        const jobs = await fetchJson('/api/jobs');
        const tbody = document.getElementById('jobsTable');
        tbody.innerHTML = '';

        if (!jobs || jobs.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5">Nenhum job</td></tr>';
          return;
        }

        for (const job of jobs) {
          const tr = document.createElement('tr');
          tr.style.cursor = 'pointer';
          tr.style.background = selectedJobId === job.job_id ? '#1e293b' : '';
          tr.innerHTML = `
            <td>${job.job_id.slice(0, 16)}</td>
            <td>${(job.model_name || '').slice(0, 15)}</td>
            <td><strong>${job.status}</strong></td>
            <td>${Math.round((job.progress || 0) * 100)}%</td>
            <td>${job.current_epoch}/${job.epochs || '?'}</td>
          `;
          tr.addEventListener('click', () => selectJob(job.job_id));
          tbody.appendChild(tr);
        }

        loadDownloads();
      } catch (e) {
        console.error('Erro ao carregar jobs:', e);
      }
    }

    async function loadDownloads() {
      try {
        const jobs = await fetchJson('/api/jobs');
        const tbody = document.getElementById('downloadsTable');
        tbody.innerHTML = '';

        const ready = (jobs || []).filter(job => job.status === 'Concluido' || job.download_path);

        if (!ready.length) {
          tbody.innerHTML = '<tr><td colspan="4">Nenhum modelo pronto para download.</td></tr>';
          document.getElementById('downloadStatus').textContent = 'Nenhum download disponível.';
          return;
        }

        for (const job of ready) {
          const tr = document.createElement('tr');
          const filename = job.download_path ? job.download_path.split('/').pop() : 'ZIP pendente';
          tr.innerHTML = `
            <td>${job.job_id.slice(0, 16)}</td>
            <td>${(job.model_name || '').slice(0, 18)}</td>
            <td>${job.status}</td>
            <td>
              ${job.download_path
                ? `<a href="/api/download-file/${encodeURIComponent(job.job_id)}" target="_blank" rel="noopener">⬇️ ${filename}</a>`
                : 'Aguardando'}
            </td>
          `;
          tbody.appendChild(tr);
        }

        if (selectedJobId) {
          const selected = ready.find(job => job.job_id === selectedJobId);
          if (selected && selected.download_path) {
            document.getElementById('downloadStatus').innerHTML =
              `Pronto para download: <strong>${selected.download_path.split('/').pop()}</strong> <br><a href="/api/download-file/${encodeURIComponent(selected.job_id)}" target="_blank" rel="noopener">Clique aqui para baixar</a>`;
          } else {
            document.getElementById('downloadStatus').textContent = 'Download ainda não está pronto.';
          }
        } else {
          document.getElementById('downloadStatus').textContent = 'Selecione um job para ver o download.';
        }
      } catch (e) {
        console.error('Erro ao carregar downloads:', e);
      }
    }

    function selectJob(jobId) {
      selectedJobId = jobId;
      document.getElementById('selectedJobId').value = jobId;
      showTab('jobs');
      loadJobs();
      reloadLogs();
      loadDownloads();
    }

    async function reloadLogs() {
      if (!selectedJobId) {
        document.getElementById('logsView').textContent = 'Selecione um job.';
        return;
      }
      try {
        const text = await fetch(`/api/logs/${encodeURIComponent(selectedJobId)}`).then(r => r.text());
        document.getElementById('logsView').textContent = text || 'Sem logs ainda.';
        document.getElementById('logsView').scrollTop = document.getElementById('logsView').scrollHeight;
      } catch (e) {
        document.getElementById('logsView').textContent = `Erro: ${e.message}`;
      }
    }

    function setTrainControlsEnabled(enabled) {
      document.getElementById('trainBtn').disabled = !enabled;
      document.getElementById('media_file').disabled = !enabled;
      document.getElementById('trainBtn').style.opacity = enabled ? '1' : '0.6';
    }

    document.getElementById('trainBtn').addEventListener('click', async () => {
      const fileInput = document.getElementById('media_file');
      const file = fileInput.files[0];
      if (!file) {
        document.getElementById('trainStatus').textContent = '❌ Selecione um arquivo.';
        return;
      }

      const form = new FormData();
      form.append('model_name', document.getElementById('model_name').value);
      form.append('author', document.getElementById('author').value);
      form.append('description', document.getElementById('description').value);
      form.append('tags', document.getElementById('tags').value);
      form.append('epochs', document.getElementById('epochs').value);
      form.append('batch_size', document.getElementById('batch_size').value);
      form.append('learning_rate', document.getElementById('learning_rate').value);
      form.append('sample_rate', document.getElementById('sample_rate').value);
      form.append('f0_method', document.getElementById('f0_method').value);
      form.append('checkpoint_interval', document.getElementById('checkpoint_interval').value);
      form.append('file', file);

      document.getElementById('trainStatus').textContent = 'Treinamento iniciado...';
      setTrainControlsEnabled(false);
      showTab('jobs');
      loadJobs();

      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 120000);

        const res = await fetch('/api/train', {
          method: 'POST',
          body: form,
          signal: controller.signal
        });

        clearTimeout(timeout);

        let json = null;
        try {
          json = await res.json();
        } catch (e) {
          const text = await res.text().catch(() => "");
          throw new Error(text || 'Resposta inválida do servidor');
        }

        if (!res.ok) {
          throw new Error(json.detail || (json.message || 'Erro ao iniciar'));
        }

        document.getElementById('trainStatus').textContent = json.message || '✅ Treinamento iniciado!';
        selectJob(json.job_id);
        fileInput.value = '';
        loadJobs();
        setTrainControlsEnabled(true);

      } catch (err) {
        const msg = err.name === 'AbortError' ? 'Timeout no upload (demorado).' : err.message || String(err);
        document.getElementById('trainStatus').textContent = `❌ ${msg}`;
        console.error('Falha ao iniciar treino:', err);
        setTrainControlsEnabled(true);
      }
    });

    document.getElementById('reloadJobsBtnTraining').addEventListener('click', () => {
      loadJobs();
      document.getElementById('trainStatus').textContent = 'Jobs recarregados.';
      setTimeout(() => {
        document.getElementById('trainStatus').textContent = 'Aguardando...';
      }, 1500);
    });

    document.getElementById('reloadLogsBtn').addEventListener('click', reloadLogs);

    document.addEventListener('DOMContentLoaded', () => {
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => showTab(btn.dataset.tab));
      });
      loadJobs();
      loadDownloads();
      setInterval(() => {
        loadJobs();
        loadDownloads();
      }, 3000);
    });
  </script>
</body>
</html>'''

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/api/jobs")
async def api_jobs():
    jobs = get_jobs()
    result = []
    for job_id, job in sorted(
        jobs.items(),
        key=lambda kv: kv[1].get("created_at", ""),
        reverse=True,
    ):
        result.append({
            "job_id": job_id,
            "model_name": job.get("model_name", ""),
            "status": job.get("status", "?"),
            "progress": float(job.get("progress", 0.0)),
            "current_epoch": int(job.get("current_epoch", 0)),
            "epochs": int(job.get("epochs", 0)),
            "message": job.get("message", ""),
            "download_path": job.get("download_path"),
        })
    return result


@app.post("/api/train")
async def api_train(
    model_name: str = Form("Modelo"),
    author: str = Form(""),
    description: str = Form(""),
    tags: str = Form(""),
    epochs: int = Form(100),
    batch_size: int = Form(16),
    learning_rate: float = Form(2e-4),
    sample_rate: str = Form("40k"),
    f0_method: str = Form("yin"),
    checkpoint_interval: int = Form(10),
    file: UploadFile = File(...),
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Selecione um arquivo.")

    epochs = max(MIN_EPOCHS, min(int(epochs), MAX_EPOCHS))
    batch_size = max(1, min(int(batch_size), 128))
    learning_rate = max(1e-6, min(float(learning_rate), 1e-2))
    checkpoint_interval = max(1, min(int(checkpoint_interval), epochs))

    ext = Path(file.filename).suffix.lower()
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"}

    if ext not in video_exts | audio_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado: {ext}",
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = TEMP_DIR / (
        f"upload_{int(time.time())}_{random.randint(1000, 9999)}{ext}"
    )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")

        temp_file.write_bytes(content)
    except HTTPException:
        temp_file.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temp_file.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao salvar arquivo: {exc}",
        ) from exc
    finally:
        await file.close()

    audio_duration = get_media_duration(str(temp_file))
    if audio_duration is not None and audio_duration < MIN_AUDIO_SECONDS:
        temp_file.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                "Áudio muito curto para treino RVC. "
                f"Mínimo recomendado: {MIN_AUDIO_SECONDS / 60:.0f} minutos."
            ),
        )

    # Deve ficar fora do bloco que rejeita áudio curto.
    job_id = generate_job_id()

    start_training_job(
        job_id=job_id,
        model_name=model_name or "Modelo",
        author=author or "Desconhecido",
        description=description or "",
        tags=tags or "",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sample_rate=sample_rate,
        f0_method=f0_method,
        checkpoint_interval=checkpoint_interval,
        audio_path=str(temp_file),
        photo_path=None,
    )

    return {
        "status": "ok",
        "job_id": job_id,
        "message": f"Treinamento iniciado. Job ID: {job_id}",
    }

@app.get("/api/logs/{job_id}")
async def api_logs(job_id: str):
    log_path = LOGS_DIR / f"{job_id}.log"
    if not log_path.exists():
        return ""
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()
    return "\n".join(lines[-100:])


@app.get("/api/job/{job_id}")
async def api_get_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return job

@app.get("/api/download-file/{job_id}")
async def api_download_file(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")

    download_path = job.get("download_path")
    if not download_path:
        raise HTTPException(status_code=404, detail="Download não pronto.")

    path = Path(download_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")

    return FileResponse(
        path,
        filename=path.name,
        media_type="application/zip",
    )


@app.delete("/api/jobs/{job_id}")
async def api_delete_job(job_id: str):
    if job_id not in get_jobs():
        raise HTTPException(status_code=404, detail=f"Job {job_id} não encontrado.")

    delete_job(job_id)
    try:
        import glob
        patterns = [
            str(MODELS_DIR / f"*_{job_id}"),
            str(CHECKPOINTS_DIR / job_id),
            str(WEIGHTS_DIR / job_id),
        ]

        for pattern in patterns:
            for path in glob.glob(pattern):
                shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

    return {"status": "ok", "job_id": job_id}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)