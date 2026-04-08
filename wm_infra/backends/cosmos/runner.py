"""Runner abstraction for Cosmos Predict jobs."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from wm_infra.controlplane import CosmosTaskConfig, ProduceSampleRequest, RolloutTaskConfig, SampleSpec, TaskType


def _guess_mime(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def _read_reference_value(reference: str) -> str:
    if reference.startswith(("http://", "https://", "data:")):
        return reference
    path = Path(reference[7:] if reference.startswith("file://") else reference)
    mime_type = _guess_mime(path)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


@dataclass(slots=True)
class CosmosRunResult:
    """Result of one Cosmos runner invocation."""

    mode: str
    output_path: str
    elapsed_s: float
    model_name: str
    seed: int | None
    frame_count: int
    fps: int
    width: int | None
    height: int | None
    error: str | None = None
    response_payload: dict[str, Any] = field(default_factory=dict)
    command: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class CosmosRunner:
    """Runner that supports Cosmos NIM, shell-based local execution, or stub mode."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        shell_runner: str | None = None,
        timeout_s: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model_name = model_name or "cosmos-predict1-7b-video2world"
        self.shell_runner = shell_runner
        self.timeout_s = timeout_s
        self._mode: str | None = None

    @property
    def mode(self) -> str:
        return self._mode or "unloaded"

    def load(self) -> str:
        if self.base_url:
            self._mode = "nim"
        elif self.shell_runner:
            self._mode = "shell"
        else:
            self._mode = "stub"
        return self._mode

    def run(
        self,
        *,
        output_dir: str | Path,
        request: ProduceSampleRequest,
        task_config: RolloutTaskConfig,
        cosmos_config: CosmosTaskConfig,
    ) -> CosmosRunResult:
        mode = self.load()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "sample.mp4"
        started = time.perf_counter()
        if mode == "nim":
            result = self._run_nim(output_path, request, task_config, cosmos_config)
        elif mode == "shell":
            result = self._run_shell(output_path, request, task_config, cosmos_config)
        else:
            result = self._run_stub(output_path, request, task_config, cosmos_config)
        result.elapsed_s = round(time.perf_counter() - started, 6)
        return result

    def _effective_seed(self, request: ProduceSampleRequest, cosmos_config: CosmosTaskConfig) -> int | None:
        if cosmos_config.seed is not None:
            return cosmos_config.seed
        return request.sample_spec.seed

    def _reference_payload(self, request: ProduceSampleRequest) -> tuple[str | None, str | None]:
        if not request.sample_spec.references:
            return None, None
        value = _read_reference_value(request.sample_spec.references[0])
        if request.task_type == TaskType.IMAGE_TO_VIDEO:
            return "image", value
        return "video", value

    def _nim_payload(
        self,
        request: ProduceSampleRequest,
        task_config: RolloutTaskConfig,
        cosmos_config: CosmosTaskConfig,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": request.sample_spec.prompt or "",
        }
        seed = self._effective_seed(request, cosmos_config)
        if seed is not None:
            payload["seed"] = seed
        if task_config.num_steps:
            payload["steps"] = task_config.num_steps
        if cosmos_config.guidance_scale:
            payload["guidance_scale"] = cosmos_config.guidance_scale
        if request.sample_spec.negative_prompt or cosmos_config.negative_prompt:
            payload["negative_prompt"] = cosmos_config.negative_prompt or request.sample_spec.negative_prompt

        media_field, media_value = self._reference_payload(request)
        if media_field and media_value:
            payload[media_field] = media_value

        video_params: dict[str, Any] = {}
        if task_config.width is not None:
            video_params["width"] = task_config.width
        if task_config.height is not None:
            video_params["height"] = task_config.height
        if task_config.frame_count is not None:
            video_params["frames_count"] = task_config.frame_count
        if cosmos_config.frames_per_second:
            video_params["frames_per_sec"] = cosmos_config.frames_per_second
        if video_params:
            payload["video_params"] = video_params
        return payload

    def _run_nim(
        self,
        output_path: Path,
        request: ProduceSampleRequest,
        task_config: RolloutTaskConfig,
        cosmos_config: CosmosTaskConfig,
    ) -> CosmosRunResult:
        assert self.base_url is not None
        payload = self._nim_payload(request, task_config, cosmos_config)
        req = urlrequest.Request(
            f"{self.base_url}/v1/infer",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urlerror.URLError as exc:
            return CosmosRunResult(
                mode="nim",
                output_path=str(output_path),
                elapsed_s=0.0,
                model_name=self.model_name,
                seed=self._effective_seed(request, cosmos_config),
                frame_count=task_config.frame_count or 0,
                fps=cosmos_config.frames_per_second,
                width=task_config.width,
                height=task_config.height,
                error=str(exc),
                response_payload={},
            )

        b64_video = response_payload.get("b64_video")
        if not b64_video:
            return CosmosRunResult(
                mode="nim",
                output_path=str(output_path),
                elapsed_s=0.0,
                model_name=self.model_name,
                seed=self._effective_seed(request, cosmos_config),
                frame_count=task_config.frame_count or 0,
                fps=cosmos_config.frames_per_second,
                width=task_config.width,
                height=task_config.height,
                error="Cosmos NIM response did not contain b64_video",
                response_payload=response_payload,
            )
        output_path.write_bytes(base64.b64decode(b64_video))
        return CosmosRunResult(
            mode="nim",
            output_path=str(output_path),
            elapsed_s=0.0,
            model_name=self.model_name,
            seed=self._effective_seed(request, cosmos_config),
            frame_count=task_config.frame_count or 0,
            fps=cosmos_config.frames_per_second,
            width=task_config.width,
            height=task_config.height,
            response_payload=response_payload,
        )

    def _run_shell(
        self,
        output_path: Path,
        request: ProduceSampleRequest,
        task_config: RolloutTaskConfig,
        cosmos_config: CosmosTaskConfig,
    ) -> CosmosRunResult:
        assert self.shell_runner is not None
        reference = request.sample_spec.references[0] if request.sample_spec.references else ""
        command = self.shell_runner.format(
            prompt=shlex.quote(request.sample_spec.prompt or ""),
            negative_prompt=shlex.quote(cosmos_config.negative_prompt or request.sample_spec.negative_prompt or ""),
            model=shlex.quote(request.model),
            variant=cosmos_config.variant.value,
            model_size=cosmos_config.model_size,
            seed="" if self._effective_seed(request, cosmos_config) is None else self._effective_seed(request, cosmos_config),
            num_steps=task_config.num_steps,
            frame_count="" if task_config.frame_count is None else task_config.frame_count,
            fps=cosmos_config.frames_per_second,
            width="" if task_config.width is None else task_config.width,
            height="" if task_config.height is None else task_config.height,
            reference_path=shlex.quote(reference),
            output_path=shlex.quote(str(output_path)),
        )
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        error = None
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip() or f"shell_runner exited with {completed.returncode}"
        return CosmosRunResult(
            mode="shell",
            output_path=str(output_path),
            elapsed_s=0.0,
            model_name=self.model_name,
            seed=self._effective_seed(request, cosmos_config),
            frame_count=task_config.frame_count or 0,
            fps=cosmos_config.frames_per_second,
            width=task_config.width,
            height=task_config.height,
            error=error,
            command=command,
            extra={"returncode": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]},
        )

    def _run_stub(
        self,
        output_path: Path,
        request: ProduceSampleRequest,
        task_config: RolloutTaskConfig,
        cosmos_config: CosmosTaskConfig,
    ) -> CosmosRunResult:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "prompt": request.sample_spec.prompt,
                    "references": request.sample_spec.references,
                    "model": request.model,
                    "variant": cosmos_config.variant.value,
                    "seed": self._effective_seed(request, cosmos_config),
                    "frame_count": task_config.frame_count,
                    "width": task_config.width,
                    "height": task_config.height,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).digest()
        output_path.write_bytes(b"COSMOS_STUB_VIDEO\x00" + digest)
        return CosmosRunResult(
            mode="stub",
            output_path=str(output_path),
            elapsed_s=0.0,
            model_name=self.model_name,
            seed=self._effective_seed(request, cosmos_config),
            frame_count=task_config.frame_count or 0,
            fps=cosmos_config.frames_per_second,
            width=task_config.width,
            height=task_config.height,
            extra={"bytes_written": output_path.stat().st_size},
        )
