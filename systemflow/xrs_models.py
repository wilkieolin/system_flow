"""
Resource prediction models for X-ray scattering reconstruction workloads.

These classes adapt fitted resource-model coefficients from Pty-Chi and
PtychoPINN benchmark studies into a common interface that SystemFlow
mutations can call.
"""

from dataclasses import dataclass, field
from math import ceil, log2
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_BUNDLED_MODEL_DATA = Path(__file__).resolve().parent / "xrs_model_data"


@dataclass(frozen=True)
class ReconstructionEstimate:
    """Predicted resources for one reconstruction workload."""

    latency_s: float
    energy_j: float
    avg_power_w: float
    images_per_joule: float
    ops: float | tuple[float, float]
    io_latency_s: float = 0.0
    compute_latency_s: float = 0.0
    io_energy_j: float = 0.0
    compute_energy_j: float = 0.0
    io_power_w: float = 0.0
    compute_power_w: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resource model coefficient file not found: {path}")
    return pd.read_csv(path)


def _select_one(df: pd.DataFrame, **filters) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in coefficient table")
        mask = mask & (df[column] == value)

    matches = df[mask]
    if len(matches) != 1:
        details = ", ".join(f"{k}={v!r}" for k, v in filters.items())
        raise ValueError(f"Expected one coefficient row for {details}, found {len(matches)}")
    return matches.iloc[0]


def _square_resolution(resolution: tuple[int, int]) -> int:
    y, x = resolution
    return int(round(np.sqrt(int(y) * int(x))))


class PtyChiResourceModel:
    """
    Resource model for Pty-Chi baseline reconstruction algorithms.

    Pty-Chi is modeled as a baseline solver family: predictions are selected
    by GPU and algorithm.
    """

    backend = "ptychi"
    default_gpu = "A100"
    default_algorithm = "dm"
    default_batch_size = 1000
    default_iterations = 500

    def __init__(
        self,
        flops_assumptions: str | Path,
        reconstruction_flops_coefficients: str | Path,
        io_coefficients: str | Path,
        power_coefficients: str | Path,
        default_gpu: str | None = None,
        default_algorithm: str | None = None,
        default_batch_size: int | None = None,
        default_iterations: int | None = None,
    ) -> None:
        self.flops_assumptions = _read_csv(flops_assumptions)
        self.reconstruction_flops_coefficients = _read_csv(reconstruction_flops_coefficients)
        self.io_coefficients = _read_csv(io_coefficients)
        self.power_coefficients = _read_csv(power_coefficients)
        if default_gpu is not None:
            self.default_gpu = default_gpu
        if default_algorithm is not None:
            self.default_algorithm = default_algorithm
        if default_batch_size is not None:
            self.default_batch_size = int(default_batch_size)
        if default_iterations is not None:
            self.default_iterations = int(default_iterations)

    @classmethod
    def from_modeling_dir(cls, modeling_dir: str | Path) -> "PtyChiResourceModel":
        modeling_dir = Path(modeling_dir)
        return cls(
            flops_assumptions=modeling_dir / "ptychi_flops_assumptions.csv",
            reconstruction_flops_coefficients=modeling_dir
            / "ptychi_reconstruction_flops_coefficients.csv",
            io_coefficients=modeling_dir / "ptychi_io_algorithm_coefficients.csv",
            power_coefficients=modeling_dir / "ptychi_power_coefficients.csv",
        )

    @classmethod
    def from_bundled_data(cls) -> "PtyChiResourceModel":
        return cls.from_modeling_dir(_BUNDLED_MODEL_DATA / "ptychi")

    def flops_per_image_epoch_gflops(
        self, algorithm: str, resolution: tuple[int, int]
    ) -> float:
        row = _select_one(self.flops_assumptions, algorithm=algorithm)
        n = _square_resolution(resolution)
        pixels = int(resolution[0]) * int(resolution[1])

        q_fft = float(row["q_fft"])
        q_elem = float(row["q_elem"])
        return (q_fft * 5.0 * pixels * log2(max(pixels, 2)) + q_elem * pixels) / 1e9

    def predict(
        self,
        num_images: int,
        resolution: tuple[int, int],
        iterations: int | None = None,
        gpu: str | None = None,
        algorithm: str | None = None,
        batch_size: int | None = None,
        include_io: bool = True,
    ) -> ReconstructionEstimate:
        gpu = self.default_gpu if gpu is None else gpu
        algorithm = self.default_algorithm if algorithm is None else algorithm
        batch_size = self.default_batch_size if batch_size is None else batch_size
        iterations = self.default_iterations if iterations is None else iterations
        num_images = int(num_images)
        iterations = int(iterations)
        batch_size = int(batch_size)
        num_batches = max(1, ceil(num_images / batch_size))
        input_pixels = num_images * int(resolution[0]) * int(resolution[1])

        recon_row = _select_one(
            self.reconstruction_flops_coefficients,
            gpu=gpu,
            algorithm=algorithm,
            batch_size=batch_size,
        )
        power_row = _select_one(self.power_coefficients, gpu=gpu, algorithm=algorithm)

        f_image_epoch = self.flops_per_image_epoch_gflops(algorithm, resolution)
        total_gflops = f_image_epoch * num_images * iterations
        compute_latency_s = (
            float(recon_row["A_recon_intercept_s"])
            + float(recon_row["s_per_gflop"]) * total_gflops
        )

        io_latency_s = 0.0
        io_power_w = 0.0
        if include_io:
            try:
                io_row = _select_one(self.io_coefficients, gpu=gpu, algorithm=algorithm)
            except ValueError:
                io_row = _select_one(self.io_coefficients, gpu=gpu)
            io_latency_s = (
                float(io_row["A_IO_intercept_s"])
                + float(io_row["beta_IO_s_per_pixel"]) * input_pixels
            )
            io_power_w = float(power_row["P_IO_w_median"])

        compute_power_w = (
            float(power_row["P_recon_intercept_w"])
            + float(power_row["P_recon_slope_w_per_epoch_batch"]) * iterations * num_batches
        )

        io_energy_j = io_latency_s * io_power_w
        compute_energy_j = compute_latency_s * compute_power_w
        latency_s = io_latency_s + compute_latency_s
        energy_j = io_energy_j + compute_energy_j
        avg_power_w = energy_j / latency_s if latency_s > 0.0 else 0.0
        images_per_joule = num_images / energy_j if energy_j > 0.0 else 0.0

        return ReconstructionEstimate(
            latency_s=latency_s,
            energy_j=energy_j,
            avg_power_w=avg_power_w,
            images_per_joule=images_per_joule,
            ops=total_gflops * 1e9,
            io_latency_s=io_latency_s,
            compute_latency_s=compute_latency_s,
            io_energy_j=io_energy_j,
            compute_energy_j=compute_energy_j,
            io_power_w=io_power_w,
            compute_power_w=compute_power_w,
            metadata={
                "backend": self.backend,
                "gpu": gpu,
                "algorithm": algorithm,
                "num_images": num_images,
                "resolution": resolution,
                "iterations": iterations,
                "batch_size": batch_size,
                "num_batches": num_batches,
                "total_gflops": total_gflops,
            },
        )


class PtychoPINNResourceModel:
    """
    Resource model for PtychoPINN inference workloads.

    PtychoPINN is modeled as one learned inference model per GPU machine, not
    as an algorithm inside the Pty-Chi baseline family.
    """

    backend = "ptychopinn"
    default_gpu = "A100"
    default_batch_size = 1024

    def __init__(
        self,
        inference_flops_coefficients: str | Path,
        io_coefficients: str | Path,
        power_coefficients: str | Path,
        default_gpu: str | None = None,
        default_batch_size: int | None = None,
    ) -> None:
        self.inference_flops_coefficients = _read_csv(inference_flops_coefficients)
        self.io_coefficients = _read_csv(io_coefficients)
        self.power_coefficients = _read_csv(power_coefficients)
        if default_gpu is not None:
            self.default_gpu = default_gpu
        if default_batch_size is not None:
            self.default_batch_size = int(default_batch_size)

    @classmethod
    def from_modeling_dir(cls, modeling_dir: str | Path) -> "PtychoPINNResourceModel":
        modeling_dir = Path(modeling_dir)
        return cls(
            inference_flops_coefficients=modeling_dir
            / "modeling_inference_flops_coefficients.csv",
            io_coefficients=modeling_dir / "modeling_io_coefficients.csv",
            power_coefficients=modeling_dir / "modeling_power_model_coefficients.csv",
        )

    @classmethod
    def from_bundled_data(cls) -> "PtychoPINNResourceModel":
        return cls.from_modeling_dir(_BUNDLED_MODEL_DATA / "ptychopinn")

    def predict(
        self,
        num_images: int,
        resolution: tuple[int, int],
        gpu: str | None = None,
        batch_size: int | None = None,
        grouped_samples: int | None = None,
        valid_ratio: float | None = 0.78,
        include_io: bool = True,
    ) -> ReconstructionEstimate:
        gpu = self.default_gpu if gpu is None else gpu
        batch_size = self.default_batch_size if batch_size is None else batch_size
        num_images = int(num_images)
        batch_size = int(batch_size)

        flops_row = _select_one(
            self.inference_flops_coefficients,
            gpu=gpu,
            batch_size=batch_size,
        )
        power_row = _select_one(self.power_coefficients, gpu=gpu)

        if grouped_samples is None:
            # PtychoPINN preprocessing groups valid samples; benchmarks are about 0.78 grouped/raw.
            ratio = 1.0 if valid_ratio is None else float(valid_ratio)
            grouped_samples = int(round(num_images * ratio))
        grouped_samples = max(1, int(grouped_samples))
        num_batches = max(1, ceil(grouped_samples / batch_size))

        baseline_n = int(flops_row["baseline_N"])
        f_sample_gflops = float(flops_row["F_sample_gflops"])
        n = _square_resolution(resolution)
        resolution_scale = (n / baseline_n) ** 2
        total_gflops = grouped_samples * f_sample_gflops * resolution_scale

        compute_latency_s = (
            float(flops_row["A_flops_intercept_s"])
            + float(flops_row["s_per_gflop"]) * total_gflops
        )

        io_latency_s = 0.0
        io_power_w = 0.0
        if include_io:
            io_row = _select_one(self.io_coefficients, gpu=gpu)
            io_latency_s = (
                float(io_row["io_intercept_s"])
                + float(io_row["io_slope_s_per_grouped"]) * grouped_samples
            )
            io_latency_s = max(0.0, io_latency_s)
            io_power_w = float(power_row["P_IO_w_median"])

        compute_power_w = (
            float(power_row["P_infer_intercept_w"])
            + float(power_row["P_infer_slope_w_per_batch"]) * num_batches
        )

        io_energy_j = io_latency_s * io_power_w
        compute_energy_j = compute_latency_s * compute_power_w
        latency_s = io_latency_s + compute_latency_s
        energy_j = io_energy_j + compute_energy_j
        avg_power_w = energy_j / latency_s if latency_s > 0.0 else 0.0
        images_per_joule = num_images / energy_j if energy_j > 0.0 else 0.0

        return ReconstructionEstimate(
            latency_s=latency_s,
            energy_j=energy_j,
            avg_power_w=avg_power_w,
            images_per_joule=images_per_joule,
            ops=total_gflops * 1e9,
            io_latency_s=io_latency_s,
            compute_latency_s=compute_latency_s,
            io_energy_j=io_energy_j,
            compute_energy_j=compute_energy_j,
            io_power_w=io_power_w,
            compute_power_w=compute_power_w,
            metadata={
                "backend": self.backend,
                "gpu": gpu,
                "num_images": num_images,
                "grouped_samples": grouped_samples,
                "resolution": resolution,
                "batch_size": batch_size,
                "num_batches": num_batches,
                "total_gflops": total_gflops,
            },
        )
