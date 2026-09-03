"""Build the self-contained Kaggle UIE-22K EvidenceMem v4 notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "EvidenceMem_Colab_T4.ipynb"
OUTPUT_NOTEBOOK = ROOT / "notebooks" / "EvidenceMem_UIE22K_Development_T4.ipynb"
CELL_DIR = ROOT / "scripts" / "notebook_cells"


def source(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(value)}


def code(value: str, *tags: str) -> dict:
    metadata = {"tags": list(tags)} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source(value),
    }


cells = [
    markdown(
        r'''
        # EvidenceMem v4 UIE-22K: reliable class-conditional visual memory

        This notebook turns the completed 224-vs-512 stress test into the next
        confirmable EvidenceMem experiment on a fixed, balanced subset of the 130k
        Universal Image Embeddings image collection.

        **Claims tested.** Does scale-corrected prototype reliability improve over
        coverage-only selection at an equal memory count? Does class-conditional
        scoring reduce salient-object shortcuts? Does continuous reliability-aware
        visual-text fusion improve accuracy, calibration, and selective risk? Which
        natively matched frozen encoder gives the strongest result on a T4?

        Development and confirmatory evaluation are separated. The default run uses
        the development split to choose the method and encoder. A final run must set a
        frozen encoder key and switch to the untouched confirmatory split. All
        predictions, selected settings, paired tests, calibration summaries,
        manifests, runtimes, and qualitative evidence are exported.
        '''
    ),
    markdown(
        r'''
        ## Before running on Kaggle

        1. In **Notebook options**, select a **GPU T4 x2** or **GPU T4** accelerator.
           The code uses one GPU so that runs are comparable with Colab T4.
        2. Turn **Internet on**. The first cell clones the public EvidenceMem branch
           and installs only missing packages.
        3. Add this Kaggle **dataset** as an input:
           `rhtsingh/130k-images-512x512-universal-image-embeddings`.
           Kaggle may also suggest the notebook
           `rhtsingh/google-universal-image-embedding-convnext-train`; its output
           contains a ConvNeXt checkpoint and logs, not the raw images, so it cannot
           replace the dataset input.
        4. Use **Run All**. The paper configuration compares native 224, 336, and 384
           pixel encoders. CUDA out-of-memory errors automatically reduce batch size,
           and embedding caches make later method runs inexpensive.

        If an earlier Kaggle version failed after finishing one or more encoders, add
        that notebook version's output as an input. This notebook will reuse a cache
        only after checking its hashes, sample IDs, split, encoder, and preprocessing
        contract.

        The 14 GB input stays read-only under `/kaggle/input`. The notebook hashes at
        most 2,400 candidates per class, keeps exactly 2,000 unique samples per class,
        and never copies the complete dataset into `/kaggle/working`.

        Start with `EVALUATION_STAGE="development"`. After choosing and recording the
        final encoder and method, set `EVALUATION_STAGE="confirmatory"` and
        `CONFIRMATORY_ENCODER_KEY` to the frozen encoder key. Do not use confirmatory
        results for further tuning. For a quick pipeline check, use `RUN_MODE="smoke"`;
        smoke results are not paper evidence.
        '''
    ),
    code(
        r'''
        # Robust bootstrap: use a local checkout when present, otherwise clone the verified branch.
        import base64
        import importlib.util
        import os
        import subprocess
        import sys
        from importlib import metadata as importlib_metadata
        from pathlib import Path

        from packaging.requirements import Requirement

        REPOSITORY_URL = "https://github.com/Prathmesh333/evidencemem.git"
        REPOSITORY_REF = os.environ.get("EVIDENCEMEM_REF", "codex/artifact-hardening")

        if Path("/kaggle/working").is_dir():
            RUNTIME_ROOT = Path("/kaggle/working")
        elif Path("/content").is_dir():
            RUNTIME_ROOT = Path("/content")
        else:
            RUNTIME_ROOT = Path.cwd()

        local_checkout = Path.cwd()
        if (
            (local_checkout / "pyproject.toml").is_file()
            and (local_checkout / "src" / "evidencemem" / "__init__.py").is_file()
        ):
            REPO_DIR = local_checkout
            USING_LOCAL_CHECKOUT = True
        else:
            REPO_DIR = RUNTIME_ROOT / "evidencemem-source"
            USING_LOCAL_CHECKOUT = False


        def get_github_token():
            token = os.environ.get("GITHUB_TOKEN", "").strip()
            if token:
                return token
            try:
                from google.colab import userdata

                return (userdata.get("GITHUB_TOKEN") or "").strip()
            except Exception:
                pass
            try:
                from kaggle_secrets import UserSecretsClient

                return (UserSecretsClient().get_secret("GITHUB_TOKEN") or "").strip()
            except Exception:
                return ""


        def authenticated_git_environment():
            environment = os.environ.copy()
            github_token = get_github_token()
            if github_token:
                credentials = base64.b64encode(
                    f"x-access-token:{github_token}".encode()
                ).decode()
                environment.update(
                    {
                        "GIT_CONFIG_COUNT": "1",
                        "GIT_CONFIG_KEY_0": "http.extraHeader",
                        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credentials}",
                    }
                )
            return environment


        def checked_command(command, *, cwd=None, environment=None, purpose):
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                details = (completed.stderr or completed.stdout or "unknown error").strip()
                raise RuntimeError(f"{purpose} failed. Command output:\n{details}")
            return completed


        if not USING_LOCAL_CHECKOUT:
            git_environment = authenticated_git_environment()
            if (REPO_DIR / ".git").is_dir():
                checked_command(
                    ["git", "fetch", "--depth", "1", "origin", REPOSITORY_REF],
                    cwd=REPO_DIR,
                    environment=git_environment,
                    purpose=f"Fetching EvidenceMem ref {REPOSITORY_REF}",
                )
                checked_command(
                    ["git", "checkout", "--detach", "FETCH_HEAD"],
                    cwd=REPO_DIR,
                    environment=git_environment,
                    purpose=f"Checking out EvidenceMem ref {REPOSITORY_REF}",
                )
            elif REPO_DIR.exists():
                raise RuntimeError(
                    f"{REPO_DIR} exists but is not a Git checkout. Remove only that "
                    "directory or start a fresh Kaggle session."
                )
            else:
                try:
                    checked_command(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "--branch",
                            REPOSITORY_REF,
                            REPOSITORY_URL,
                            str(REPO_DIR),
                        ],
                        environment=git_environment,
                        purpose=f"Cloning EvidenceMem ref {REPOSITORY_REF}",
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        f"{error}\nTurn Kaggle Internet on and rerun this cell. "
                        "The configured branch is public and does not require a token."
                    ) from error

        REQUIRED = {
            "open_clip": "open_clip_torch>=2.30,<4",
            "faiss": "faiss-cpu>=1.8,<2",
            "imagehash": "ImageHash>=4.3,<5",
            "sklearn": "scikit-learn>=1.4,<2",
            "pandas": "pandas>=2.0,<4",
            "seaborn": "seaborn>=0.13,<1",
            "scipy": "scipy>=1.11,<2",
            "tqdm": "tqdm>=4.66,<5",
            "psutil": "psutil>=5.9,<8",
            "transformers": "transformers>=4.50,<6",
            "yaml": "PyYAML>=6,<7",
        }
        def requirement_is_unsatisfied(module, specification):
            if importlib.util.find_spec(module) is None:
                return True
            requirement = Requirement(specification)
            try:
                installed = importlib_metadata.version(requirement.name)
            except importlib_metadata.PackageNotFoundError:
                return True
            return installed not in requirement.specifier


        missing = [
            specification
            for module, specification in REQUIRED.items()
            if requirement_is_unsatisfied(module, specification)
        ]
        if missing:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", *missing]
            )

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)]
        )
        SOURCE_DIR = REPO_DIR / "src"
        if not (SOURCE_DIR / "evidencemem" / "__init__.py").is_file():
            raise RuntimeError(f"EvidenceMem source package is missing from {SOURCE_DIR}")
        source_path = str(SOURCE_DIR.resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        importlib.invalidate_caches()
        for module_name in list(sys.modules):
            if module_name == "evidencemem" or module_name.startswith("evidencemem."):
                del sys.modules[module_name]

        import evidencemem as _evidencemem
        from evidencemem import SelectionConfig as _SelectionConfig
        from evidencemem.benchmark import fit_prototype_memory as _fit_prototype_memory
        from evidencemem.cache import save_embedding_cache as _save_embedding_cache

        if _evidencemem.__version__ != "0.2.0":
            raise RuntimeError(
                f"Expected EvidenceMem 0.2.0, imported {_evidencemem.__version__} "
                f"from {_evidencemem.__file__}"
            )
        print(
            f"Bootstrap verified: EvidenceMem {_evidencemem.__version__} "
            f"from {_evidencemem.__file__}"
        )
        print("Runtime root:", RUNTIME_ROOT)
        ''',
        "bootstrap",
    ),
    code(
        r'''
        # Configuration: paper mode is the predeclared EvidenceMem v4 protocol.
        from dataclasses import asdict, dataclass

        RUN_MODE = "paper"  # use "smoke" only to verify the pipeline
        EVALUATION_STAGE = os.environ.get(
            "EVIDENCEMEM_EVALUATION_STAGE", "development"
        ).strip().lower()
        CONFIRMATORY_ENCODER_KEY = os.environ.get(
            "EVIDENCEMEM_CONFIRMATORY_ENCODER", ""
        ).strip()
        # Usually leave this empty. Set UIE_DATASET_ROOT only for a custom mount.
        DATASET_ROOT_OVERRIDE = os.environ.get("UIE_DATASET_ROOT", "").strip()
        RAW_DATASET_SLUG = "rhtsingh/130k-images-512x512-universal-image-embeddings"
        RAW_DATASET_MOUNT_CANDIDATES = (
            "/kaggle/input/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings",
            "/kaggle/input/130k-images-512x512-universal-image-embeddings",
        )
        CONVNEXT_NOTEBOOK_OUTPUT = (
            "/kaggle/input/notebooks/rhtsingh/"
            "google-universal-image-embedding-convnext-train"
        )
        PROTOCOL_ID = "uie22k_evidencemem_v4"
        PROTOCOL_REVISION = "2.0.1"

        PAPER_ENCODERS = (
            (
                "clip_b32_224",
                "open_clip",
                "ViT-B-32-quickgelu",
                "openai",
                224,
                128,
            ),
            (
                "clip_b16_224",
                "open_clip",
                "ViT-B-16-quickgelu",
                "openai",
                224,
                96,
            ),
            (
                "clip_l14_336",
                "open_clip",
                "ViT-L-14-336-quickgelu",
                "openai",
                336,
                24,
            ),
            (
                "siglip2_b16_384",
                "transformers",
                "google/siglip2-base-patch16-384",
                "google",
                384,
                16,
            ),
        )
        SMOKE_ENCODERS = (PAPER_ENCODERS[0],)

        CLASS_NAMES = (
            "apparel",
            "artwork",
            "cars",
            "dishes",
            "furniture",
            "illustrations",
            "landmark",
            "meme",
            "packaged",
            "storefronts",
            "toys",
        )
        CLASS_PROMPT_NAMES = {
            "apparel": "clothing or fashion apparel",
            "artwork": "a work of art",
            "cars": "a car",
            "dishes": "a prepared food dish",
            "furniture": "furniture",
            "illustrations": "an illustration",
            "landmark": "a landmark",
            "meme": "an internet meme",
            "packaged": "a packaged retail product",
            "storefronts": "a storefront",
            "toys": "a toy",
        }
        PROMPTS = (
            "a photo whose main category is {}.",
            "a catalog image of {}.",
            "a real-world image showing {} as the main subject.",
            "a wide-view image categorized as {}.",
            "an online image whose visual domain is {}.",
            "an image of {}, not merely a small foreground object.",
        )


        @dataclass(frozen=True)
        class RunConfig:
            mode: str
            sample_seed: int
            samples_per_class: int
            candidate_cap_per_class: int
            train_per_class: int
            val_per_class: int
            development_per_class: int
            confirmatory_per_class: int
            phash_distance: int
            hash_workers: int
            seeds: tuple
            encoders: tuple
            num_workers: int
            budgets: tuple
            default_budget: int
            topk_grid: tuple
            class_topk_grid: tuple
            alpha_grid: tuple
            temperatures: tuple
            reliability_power_grid: tuple
            reliability_weight_grid: tuple
            gate_slope_grid: tuple


        if RUN_MODE == "paper":
            CFG = RunConfig(
                mode=RUN_MODE,
                sample_seed=2026,
                samples_per_class=2_000,
                candidate_cap_per_class=2_400,
                train_per_class=1_200,
                val_per_class=200,
                development_per_class=300,
                confirmatory_per_class=300,
                phash_distance=3,
                hash_workers=8,
                seeds=(7, 17, 29, 43, 61),
                encoders=PAPER_ENCODERS,
                num_workers=0,
                budgets=(20, 40, 80),
                default_budget=40,
                topk_grid=(3, 5, 10, 20),
                class_topk_grid=(1, 3, 5, 10),
                alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
                temperatures=(0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.50, 1.0),
                reliability_power_grid=(0.0, 0.5, 1.0, 2.0),
                reliability_weight_grid=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.5, 0.5, 0.0),
                    (0.5, 0.0, 0.5),
                    (0.0, 0.5, 0.5),
                    (0.35, 0.40, 0.25),
                    (0.45, 0.35, 0.20),
                ),
                gate_slope_grid=(-0.50, -0.25, 0.0, 0.25, 0.50),
            )
        elif RUN_MODE == "smoke":
            CFG = RunConfig(
                mode=RUN_MODE,
                sample_seed=2026,
                samples_per_class=20,
                candidate_cap_per_class=30,
                train_per_class=12,
                val_per_class=2,
                development_per_class=3,
                confirmatory_per_class=3,
                phash_distance=0,
                hash_workers=4,
                seeds=(7,),
                encoders=SMOKE_ENCODERS,
                num_workers=0,
                budgets=(1, 2, 3),
                default_budget=2,
                topk_grid=(1, 2),
                class_topk_grid=(1, 2),
                alpha_grid=(0.0, 0.5, 1.0),
                temperatures=(0.05, 0.10, 0.50, 1.0),
                reliability_power_grid=(0.0, 1.0),
                reliability_weight_grid=((1.0, 0.0, 0.0), (0.35, 0.40, 0.25)),
                gate_slope_grid=(-0.5, 0.0, 0.5),
            )
        else:
            raise ValueError("RUN_MODE must be 'paper' or 'smoke'.")

        if EVALUATION_STAGE not in {"development", "confirmatory"}:
            raise ValueError(
                "EVALUATION_STAGE must be 'development' or 'confirmatory'."
            )
        if EVALUATION_STAGE == "confirmatory" and not CONFIRMATORY_ENCODER_KEY:
            raise ValueError(
                "Set EVIDENCEMEM_CONFIRMATORY_ENCODER to the frozen development "
                "winner before a confirmatory run."
            )
        if (
            CFG.train_per_class
            + CFG.val_per_class
            + CFG.development_per_class
            + CFG.confirmatory_per_class
            != CFG.samples_per_class
        ):
            raise ValueError("The per-class split sizes must sum to samples_per_class.")
        ENCODER_SPECS = {
            item[0]: {
                "key": item[0],
                "backend": item[1],
                "model": item[2],
                "weights": item[3],
                "resolution": int(item[4]),
                "batch_size": int(item[5]),
            }
            for item in CFG.encoders
        }
        if EVALUATION_STAGE == "confirmatory":
            if CONFIRMATORY_ENCODER_KEY not in ENCODER_SPECS:
                raise ValueError(
                    f"Unknown confirmatory encoder {CONFIRMATORY_ENCODER_KEY!r}; "
                    f"choose one of {sorted(ENCODER_SPECS)}."
                )
            ENCODER_SPECS = {
                CONFIRMATORY_ENCODER_KEY: ENCODER_SPECS[CONFIRMATORY_ENCODER_KEY]
            }
        EVALUATION_SPLIT = (
            "development" if EVALUATION_STAGE == "development" else "confirmatory"
        )
        CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
        N_CLASSES = len(CLASS_NAMES)
        print(asdict(CFG))
        print("Evaluation stage:", EVALUATION_STAGE)
        print("Encoders:", sorted(ENCODER_SPECS))
        print("Total retained images:", CFG.samples_per_class * N_CLASSES)
        ''',
        "parameters",
    ),
    code(
        r'''
        # Imports, deterministic execution, runtime checks, and crash-safe artifact helpers.
        import hashlib
        import io
        import json
        import math
        import platform
        import random
        import shutil
        import tempfile
        import time
        from concurrent.futures import ThreadPoolExecutor
        from datetime import UTC, datetime

        import faiss
        import imagehash
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import psutil
        import seaborn as sns
        import sklearn
        import torch
        import torch.nn.functional as F
        import torchvision
        from IPython.display import display
        from PIL import Image, ImageFile
        from evidencemem import ReliabilityWeights, SelectionConfig
        from evidencemem.artifacts import (
            atomic_write_json,
            finalize_run_manifest,
            git_revision,
            start_run_manifest,
        )
        from evidencemem.benchmark import (
            PrototypeArrays,
            class_conditional_visual_scores,
            exact_search,
            fit_prototype_memory,
            fused_class_scores,
            reweight_prototype_reliability,
            text_class_scores,
            tip_adapter_scores,
            weighted_knn_scores,
        )
        from evidencemem.cache import load_embedding_cache, save_embedding_cache
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            log_loss,
        )
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        ImageFile.LOAD_TRUNCATED_IMAGES = False
        sns.set_theme(context="notebook", style="whitegrid")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DEVICE.type != "cuda":
            raise RuntimeError(
                "CUDA is unavailable. In Kaggle Notebook options select a T4 GPU and restart."
            )
        GPU_NAME = torch.cuda.get_device_name(0)
        if "T4" not in GPU_NAME.upper():
            print(f"Warning: requested a T4, but Kaggle assigned {GPU_NAME!r}; continuing.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


        def seed_everything(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


        def normalize(values):
            matrix = np.asarray(values, dtype=np.float32)
            return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


        def atomic_json(path, payload):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            os.replace(temporary, path)


        def atomic_csv(frame, path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, path)


        SOURCE_ID = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()
        ROOT = RUNTIME_ROOT / "EvidenceMem"
        CACHE_DIR = ROOT / "cache" / PROTOCOL_ID
        RUN_DIR = ROOT / "runs" / (
            f"{CFG.mode}_{EVALUATION_STAGE}_{PROTOCOL_ID}_{SOURCE_ID}"
        )
        for directory in (CACHE_DIR, RUN_DIR):
            directory.mkdir(parents=True, exist_ok=True)


        def journal(event, **payload):
            record = {"time_utc": datetime.now(UTC).isoformat(), "event": event, **payload}
            with (RUN_DIR / "experiment_journal.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")


        environment = {
            "protocol_id": PROTOCOL_ID,
            "protocol_revision": PROTOCOL_REVISION,
            "source_id": SOURCE_ID,
            "config": asdict(CFG),
            "device": str(DEVICE),
            "gpu": GPU_NAME,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "sklearn": sklearn.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "cpu_count": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
            "evaluation_stage": EVALUATION_STAGE,
            "evaluation_split": EVALUATION_SPLIT,
            "encoder_specs": ENCODER_SPECS,
        }
        RUN_MANIFEST = start_run_manifest(
            config=asdict(CFG),
            repository=REPO_DIR,
            model_name=",".join(sorted(ENCODER_SPECS)),
            pretrained="frozen pretrained checkpoints",
        )
        atomic_write_json(RUN_DIR / "run_manifest.json", RUN_MANIFEST)
        atomic_json(RUN_DIR / "environment.json", environment)
        journal("run_started", **environment)
        environment
        ''',
        "runtime",
    ),
    markdown(
        r'''
        ## 1. Fixed UIE-22K sampling manifest

        The sampler first enumerates file names without loading image tensors. For
        each class, it hashes at most 2,400 deterministically chosen candidates,
        rejects corrupt images and cross-label duplicate groups, removes exact-byte
        duplicates, and clusters perceptual hashes within Hamming distance three.
        One representative remains from each perceptual cluster.

        Exactly 2,000 unique samples per class are then assigned to 1,200 train, 200
        validation, 300 development, and 300 confirmatory examples. The resulting CSV
        manifest is immutable input to every encoder and method. Validation selects
        method settings; development selects one encoder; confirmatory data is opened
        only after that encoder is frozen.
        '''
    ),
    code(
        r'''
        IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


        def has_expected_class_folders(directory):
            return directory.is_dir() and all(
                (directory / class_name).is_dir() for class_name in CLASS_NAMES
            )


        def bounded_directories(root, maximum_depth=4):
            """Yield directories breadth-first without descending into a valid image root."""
            root = Path(root)
            if not root.is_dir():
                return
            queue = [(root, 0)]
            seen = set()
            while queue:
                directory, depth = queue.pop(0)
                try:
                    identity = directory.resolve()
                except OSError:
                    identity = directory
                if identity in seen:
                    continue
                seen.add(identity)
                yield directory
                if depth >= maximum_depth or has_expected_class_folders(directory):
                    continue
                try:
                    children = sorted(
                        child for child in directory.iterdir() if child.is_dir()
                    )
                except (OSError, PermissionError):
                    continue
                queue.extend((child, depth + 1) for child in children)


        def directory_preview(directory, limit=8):
            try:
                names = sorted(path.name for path in directory.iterdir())
            except (OSError, PermissionError):
                return "unreadable"
            shown = names[:limit]
            suffix = f", ... (+{len(names) - limit} more)" if len(names) > limit else ""
            return ", ".join(shown) + suffix if shown else "empty"


        def locate_dataset_root():
            seeds = []
            if DATASET_ROOT_OVERRIDE:
                seeds.append(Path(DATASET_ROOT_OVERRIDE))
            seeds.extend(Path(path) for path in RAW_DATASET_MOUNT_CANDIDATES)
            seeds.append(Path("/kaggle/input"))

            checked = set()
            for seed in seeds:
                for candidate in bounded_directories(seed):
                    try:
                        identity = candidate.resolve()
                    except OSError:
                        identity = candidate
                    if identity in checked:
                        continue
                    checked.add(identity)
                    if has_expected_class_folders(candidate):
                        print(f"Validated raw UIE dataset: {candidate}")
                        return candidate

            notebook_output = Path(CONVNEXT_NOTEBOOK_OUTPUT)
            wrong_input_note = ""
            if notebook_output.is_dir():
                wrong_input_note = (
                    f"\nDetected {notebook_output}, but it is the output of the ConvNeXt "
                    "training notebook, not the image dataset. Its top-level contents are: "
                    f"{directory_preview(notebook_output)}. Keep or remove that input; it is "
                    "not used by this experiment."
                )
            override_note = (
                f"\nUIE_DATASET_ROOT was set to {DATASET_ROOT_OVERRIDE!r}, but no compatible "
                "class-folder root was found below it."
                if DATASET_ROOT_OVERRIDE
                else ""
            )
            raise FileNotFoundError(
                "Could not find the raw UIE images. In Kaggle, choose Add Input -> "
                f"Datasets -> {RAW_DATASET_SLUG}. The correct input contains these 11 "
                "folders directly: "
                + ", ".join(CLASS_NAMES)
                + wrong_input_note
                + override_note
            )


        def iter_image_files(directory):
            return sorted(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )


        def stable_seed(text, seed):
            digest = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).digest()
            return int.from_bytes(digest[:8], "little", signed=False)


        def deterministic_rank(text, seed):
            return hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()


        def inspect_candidate(record):
            path = Path(record["absolute_path"])
            try:
                blob = path.read_bytes()
                content_sha256 = hashlib.sha256(blob).hexdigest()
                with Image.open(io.BytesIO(blob)) as image:
                    width, height = image.size
                    rgb = image.convert("RGB")
                    perceptual_hash = str(imagehash.phash(rgb, hash_size=8))
                return {
                    **record,
                    "status": "ok",
                    "sha256": content_sha256,
                    "phash": perceptual_hash,
                    "width": int(width),
                    "height": int(height),
                    "file_bytes": len(blob),
                    "error": "",
                }
            except Exception as error:
                return {
                    **record,
                    "status": "corrupt",
                    "sha256": "",
                    "phash": "",
                    "width": -1,
                    "height": -1,
                    "file_bytes": int(path.stat().st_size) if path.exists() else -1,
                    "error": f"{type(error).__name__}: {error}",
                }


        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [0] * size

            def find(self, item):
                while self.parent[item] != item:
                    self.parent[item] = self.parent[self.parent[item]]
                    item = self.parent[item]
                return item

            def union(self, left, right):
                left_root, right_root = self.find(left), self.find(right)
                if left_root == right_root:
                    return
                if self.rank[left_root] < self.rank[right_root]:
                    left_root, right_root = right_root, left_root
                self.parent[right_root] = left_root
                if self.rank[left_root] == self.rank[right_root]:
                    self.rank[left_root] += 1


        class BKTree:
            """BK-tree over 64-bit perceptual hashes using Hamming distance."""

            def __init__(self):
                self.root = None
                self.nodes = {}

            @staticmethod
            def distance(left, right):
                return int(left ^ right).bit_count()

            def add(self, value, index):
                if self.root is None:
                    self.root = index
                    self.nodes[index] = {"value": value, "indices": [index], "children": {}}
                    return
                node_index = self.root
                while True:
                    node = self.nodes[node_index]
                    distance = self.distance(value, node["value"])
                    if distance == 0:
                        node["indices"].append(index)
                        return
                    child = node["children"].get(distance)
                    if child is None:
                        node["children"][distance] = index
                        self.nodes[index] = {
                            "value": value,
                            "indices": [index],
                            "children": {},
                        }
                        return
                    node_index = child

            def query(self, value, maximum_distance):
                if self.root is None:
                    return []
                matches = []
                stack = [self.root]
                while stack:
                    node_index = stack.pop()
                    node = self.nodes[node_index]
                    distance = self.distance(value, node["value"])
                    if distance <= maximum_distance:
                        matches.extend(node["indices"])
                    lower = distance - maximum_distance
                    upper = distance + maximum_distance
                    stack.extend(
                        child
                        for edge, child in node["children"].items()
                        if lower <= edge <= upper
                    )
                return matches


        def manifest_checksum(frame):
            columns = ["split", "label", "label_id", "relative_path", "sha256", "phash"]
            canonical = "\n".join(
                "\0".join(str(value) for value in row)
                for row in frame[columns].itertuples(index=False, name=None)
            )
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


        def validate_manifest(frame, dataset_root):
            required = {
                "sample_id",
                "split",
                "label",
                "label_id",
                "relative_path",
                "sha256",
                "phash",
            }
            missing_columns = required.difference(frame.columns)
            if missing_columns:
                raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
            expected = {
                "train": CFG.train_per_class,
                "val": CFG.val_per_class,
                "development": CFG.development_per_class,
                "confirmatory": CFG.confirmatory_per_class,
            }
            counts = frame.groupby(["label", "split"]).size()
            for class_name in CLASS_NAMES:
                for split_name, expected_count in expected.items():
                    observed = int(counts.get((class_name, split_name), 0))
                    if observed != expected_count:
                        raise ValueError(
                            f"Manifest count mismatch for {class_name}/{split_name}: "
                            f"expected {expected_count}, observed {observed}"
                        )
            if frame["sha256"].duplicated().any():
                raise ValueError("Exact duplicate content remains in the retained manifest.")
            missing_files = [
                relative
                for relative in frame["relative_path"]
                if not (dataset_root / relative).is_file()
            ]
            if missing_files:
                raise FileNotFoundError(
                    f"{len(missing_files)} manifest files are missing; first: {missing_files[0]}"
                )


        def build_manifest(dataset_root):
            source_counts = {}
            candidate_records = []
            for class_name in CLASS_NAMES:
                files = iter_image_files(dataset_root / class_name)
                source_counts[class_name] = len(files)
                if len(files) < CFG.samples_per_class:
                    raise ValueError(
                        f"Class {class_name!r} has only {len(files)} images; "
                        f"the protocol requires {CFG.samples_per_class}."
                    )
                candidate_count = min(CFG.candidate_cap_per_class, len(files))
                rng = np.random.default_rng(stable_seed(class_name, CFG.sample_seed))
                chosen = np.sort(rng.choice(len(files), size=candidate_count, replace=False))
                for file_index in chosen:
                    path = files[int(file_index)]
                    candidate_records.append(
                        {
                            "candidate_id": len(candidate_records),
                            "label": class_name,
                            "label_id": CLASS_TO_ID[class_name],
                            "relative_path": path.relative_to(dataset_root).as_posix(),
                            "absolute_path": str(path),
                        }
                    )

            print("Source counts:", source_counts)
            print(f"Hashing {len(candidate_records):,} candidate images...")
            with ThreadPoolExecutor(max_workers=CFG.hash_workers) as executor:
                inspected = list(
                    tqdm(
                        executor.map(inspect_candidate, candidate_records),
                        total=len(candidate_records),
                        desc="SHA-256 + pHash",
                    )
                )
            audit = pd.DataFrame(inspected).set_index("candidate_id", drop=False)

            exact_representatives = []
            valid = audit[audit["status"] == "ok"]
            for _, group in valid.groupby("sha256", sort=False):
                indices = group.index.tolist()
                if group["label"].nunique() > 1:
                    audit.loc[indices, "status"] = "cross_label_exact"
                    continue
                representative = min(
                    indices,
                    key=lambda index: deterministic_rank(
                        audit.at[index, "relative_path"], CFG.sample_seed
                    ),
                )
                exact_representatives.append(representative)
                duplicates = [index for index in indices if index != representative]
                if duplicates:
                    audit.loc[duplicates, "status"] = "exact_duplicate"

            survivors = audit.loc[exact_representatives].copy().reset_index(drop=True)
            union_find = UnionFind(len(survivors))
            tree = BKTree()
            for index, phash_text in enumerate(
                tqdm(survivors["phash"], desc="Perceptual duplicate grouping")
            ):
                value = int(phash_text, 16)
                for neighbor in tree.query(value, CFG.phash_distance):
                    union_find.union(index, neighbor)
                tree.add(value, index)
            survivors["near_group"] = [union_find.find(i) for i in range(len(survivors))]

            retained_candidate_ids = []
            for _, group in survivors.groupby("near_group", sort=False):
                audit_indices = group["candidate_id"].astype(int).tolist()
                if group["label"].nunique() > 1:
                    audit.loc[audit_indices, "status"] = "cross_label_perceptual"
                    continue
                representative_row = min(
                    group.index,
                    key=lambda index: deterministic_rank(
                        survivors.at[index, "relative_path"], CFG.sample_seed + 1
                    ),
                )
                representative_id = int(survivors.at[representative_row, "candidate_id"])
                retained_candidate_ids.append(representative_id)
                duplicates = [index for index in audit_indices if index != representative_id]
                if duplicates:
                    audit.loc[duplicates, "status"] = "perceptual_duplicate"
                audit.loc[representative_id, "status"] = "unique"

            unique = audit.loc[retained_candidate_ids].copy()
            unique_counts = unique.groupby("label").size().to_dict()
            insufficient = {
                class_name: int(unique_counts.get(class_name, 0))
                for class_name in CLASS_NAMES
                if int(unique_counts.get(class_name, 0)) < CFG.samples_per_class
            }
            if insufficient:
                raise RuntimeError(
                    "Not enough unique candidates after deduplication: "
                    f"{insufficient}. Lower samples_per_class explicitly or increase the "
                    "candidate cap; the notebook will not silently change the protocol."
                )

            selected_parts = []
            for class_name in CLASS_NAMES:
                group = unique[unique["label"] == class_name].copy()
                rng = np.random.default_rng(stable_seed(class_name, CFG.sample_seed + 2))
                order = rng.permutation(len(group))[: CFG.samples_per_class]
                selected = group.iloc[order].copy().reset_index(drop=True)
                split_rng = np.random.default_rng(stable_seed(class_name, CFG.sample_seed + 3))
                split_order = split_rng.permutation(len(selected))
                split_values = np.empty(len(selected), dtype=object)
                train_end = CFG.train_per_class
                val_end = train_end + CFG.val_per_class
                development_end = val_end + CFG.development_per_class
                split_values[split_order[:train_end]] = "train"
                split_values[split_order[train_end:val_end]] = "val"
                split_values[split_order[val_end:development_end]] = "development"
                split_values[split_order[development_end:]] = "confirmatory"
                selected["split"] = split_values
                selected_parts.append(selected)

            manifest = pd.concat(selected_parts, ignore_index=True)
            manifest["sample_id"] = manifest["sha256"]
            manifest["_split_order"] = manifest["split"].map(
                {"train": 0, "val": 1, "development": 2, "confirmatory": 3}
            )
            manifest = manifest.sort_values(
                ["_split_order", "label_id", "relative_path"], kind="stable"
            ).reset_index(drop=True)
            manifest = manifest.drop(columns=["_split_order", "absolute_path", "error"])
            manifest["manifest_row"] = np.arange(len(manifest), dtype=np.int64)
            return manifest, audit.reset_index(drop=True), source_counts, unique_counts


        DATASET_ROOT = locate_dataset_root()
        sampling_tag = (
            f"s{CFG.samples_per_class}_c{CFG.candidate_cap_per_class}_"
            f"ph{CFG.phash_distance}_seed{CFG.sample_seed}"
        )
        MANIFEST_CACHE_PATH = CACHE_DIR / f"uie_manifest_{sampling_tag}.csv"
        AUDIT_CACHE_PATH = CACHE_DIR / f"uie_candidate_audit_{sampling_tag}.csv"

        if MANIFEST_CACHE_PATH.is_file():
            manifest_df = pd.read_csv(MANIFEST_CACHE_PATH)
            source_counts = {
                class_name: len(iter_image_files(DATASET_ROOT / class_name))
                for class_name in CLASS_NAMES
            }
            unique_counts = manifest_df.groupby("label").size().to_dict()
            print("Loaded cached sampling manifest:", MANIFEST_CACHE_PATH)
        else:
            manifest_df, audit_df, source_counts, unique_counts = build_manifest(DATASET_ROOT)
            atomic_csv(manifest_df, MANIFEST_CACHE_PATH)
            atomic_csv(audit_df.drop(columns=["absolute_path"]), AUDIT_CACHE_PATH)

        validate_manifest(manifest_df, DATASET_ROOT)
        MANIFEST_ID = manifest_checksum(manifest_df)
        atomic_csv(manifest_df, RUN_DIR / "uie22k_manifest.csv")
        audit_status_counts = {}
        if AUDIT_CACHE_PATH.is_file():
            audit_status_counts = (
                pd.read_csv(AUDIT_CACHE_PATH)["status"].value_counts().sort_index().to_dict()
            )
        sampling_summary = {
            "manifest_id": MANIFEST_ID,
            "dataset_root_runtime": str(DATASET_ROOT),
            "dataset_slug": RAW_DATASET_SLUG,
            "source_class_counts": {key: int(value) for key, value in source_counts.items()},
            "unique_candidate_counts": {key: int(value) for key, value in unique_counts.items()},
            "audit_status_counts": {key: int(value) for key, value in audit_status_counts.items()},
            "sampling_config": asdict(CFG),
            "retained_count": int(len(manifest_df)),
            "split_counts": {
                key: int(value) for key, value in manifest_df["split"].value_counts().items()
            },
        }
        atomic_json(RUN_DIR / "sampling_summary.json", sampling_summary)
        journal("sampling_manifest_ready", **sampling_summary)
        print("Dataset root:", DATASET_ROOT)
        print("Manifest SHA-256:", MANIFEST_ID)
        display(pd.crosstab(manifest_df["label"], manifest_df["split"]))
        ''',
        "uie-sampling",
    ),
    code(
        r'''
        # Dataset wrapper keeps sample order identical across encoders and methods.
        class ManifestImageDataset(Dataset):
            def __init__(self, frame, dataset_root, transform):
                self.frame = frame.reset_index(drop=True).copy()
                self.dataset_root = Path(dataset_root)
                self.transform = transform

            def __len__(self):
                return len(self.frame)

            def __getitem__(self, index):
                row = self.frame.iloc[index]
                path = self.dataset_root / row["relative_path"]
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    tensor = self.transform(rgb)
                return tensor, int(row["label_id"])

            @property
            def sample_ids(self):
                return self.frame["sample_id"].astype(str).to_numpy()


        SPLIT_FRAMES = {
            split_name: manifest_df[manifest_df["split"] == split_name]
            .sort_values(["label_id", "relative_path"], kind="stable")
            .reset_index(drop=True)
            for split_name in ("train", "val", "development", "confirmatory")
        }
        for split_name, frame in SPLIT_FRAMES.items():
            expected = {
                "train": CFG.train_per_class * N_CLASSES,
                "val": CFG.val_per_class * N_CLASSES,
                "development": CFG.development_per_class * N_CLASSES,
                "confirmatory": CFG.confirmatory_per_class * N_CLASSES,
            }[split_name]
            if len(frame) != expected:
                raise AssertionError(f"Unexpected {split_name} size: {len(frame)} != {expected}")
        print({name: len(frame) for name, frame in SPLIT_FRAMES.items()})
        ''',
        "uie-dataset",
    ),
    markdown(
        r'''
        ## 2. Frozen native-resolution encoder sweep

        The completed stress test showed that forcing the 224px ViT-B/32 checkpoint to
        512px hurt every method. This experiment instead compares checkpoints at their
        native resolutions: OpenAI CLIP B/32 and B/16 at 224, CLIP L/14 at 336, and
        SigLIP 2 B/16 at 384. Each encoder is loaded, evaluated, and released before the
        next one is created so the sweep fits on a single T4. Caches are keyed by the
        immutable manifest, encoder identity, split, preprocessing contract, and source
        revision.
        '''
    ),
    code(
        r'''
        import open_clip
        from transformers import AutoModel, AutoProcessor

        OPENCLIP_MODELS = set(open_clip.list_models())
        OPENCLIP_PRETRAINED = set(open_clip.list_pretrained())
        PREPROCESS_CONTRACTS = {
            "open_clip": "open_clip_native_eval_v1",
            "transformers": "hf_slow_processor_native_eval_v2",
        }


        def pooled_feature_tensor(output, *, source):
            """Accept tensor and structured Hugging Face feature-return APIs."""
            if torch.is_tensor(output):
                return output
            pooled = getattr(output, "pooler_output", None)
            if torch.is_tensor(pooled):
                return pooled
            if isinstance(output, (tuple, list)):
                for candidate in output:
                    if torch.is_tensor(candidate) and candidate.ndim == 2:
                        return candidate
            raise TypeError(
                f"{source} returned {type(output).__name__} without a pooled tensor."
            )


        def load_encoder(spec):
            if spec["backend"] == "open_clip":
                if spec["model"] not in OPENCLIP_MODELS:
                    raise RuntimeError(
                        f"OpenCLIP does not provide model {spec['model']!r}."
                    )
                if (spec["model"], spec["weights"]) not in OPENCLIP_PRETRAINED:
                    raise RuntimeError(
                        f"OpenCLIP does not provide {spec['weights']!r} weights "
                        f"for {spec['model']!r}."
                    )
                model, _, preprocess = open_clip.create_model_and_transforms(
                    spec["model"], pretrained=spec["weights"], device=DEVICE
                )
                model.eval().requires_grad_(False)
                configured = getattr(model.visual, "image_size", spec["resolution"])
                if isinstance(configured, (tuple, list)):
                    configured = configured[0]
                if int(configured) != int(spec["resolution"]):
                    raise RuntimeError(
                        f"Encoder {spec['key']} reports image size {configured}, "
                        f"expected {spec['resolution']}."
                    )
                tokenizer = open_clip.get_tokenizer(spec["model"])

                def encode_images(images):
                    return model.encode_image(images)

                def encode_texts(texts):
                    return model.encode_text(tokenizer(texts).to(DEVICE))

                return {
                    "model": model,
                    "preprocess": preprocess,
                    "encode_images": encode_images,
                    "encode_texts": encode_texts,
                }

            if spec["backend"] == "transformers":
                try:
                    processor = AutoProcessor.from_pretrained(
                        spec["model"], use_fast=False
                    )
                except TypeError:
                    processor = AutoProcessor.from_pretrained(spec["model"])
                try:
                    model = AutoModel.from_pretrained(
                        spec["model"], dtype=torch.float16
                    )
                except TypeError:
                    model = AutoModel.from_pretrained(
                        spec["model"], torch_dtype=torch.float16
                    )
                model = model.to(DEVICE).eval().requires_grad_(False)

                def preprocess(image):
                    return processor(images=image, return_tensors="pt")[
                        "pixel_values"
                    ][0]

                def encode_images(images):
                    output = model.get_image_features(pixel_values=images)
                    return pooled_feature_tensor(
                        output, source="get_image_features"
                    )

                def encode_texts(texts):
                    inputs = processor(
                        text=texts,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_inputs = {
                        key: value.to(DEVICE)
                        for key, value in inputs.items()
                        if key in {"input_ids", "attention_mask", "position_ids"}
                    }
                    output = model.get_text_features(**text_inputs)
                    return pooled_feature_tensor(
                        output, source="get_text_features"
                    )

                return {
                    "model": model,
                    "processor": processor,
                    "preprocess": preprocess,
                    "encode_images": encode_images,
                    "encode_texts": encode_texts,
                }
            raise ValueError(f"Unknown encoder backend: {spec['backend']}")


        def encode_text_prototypes(adapter):
            vectors = []
            with torch.inference_mode():
                for class_name in CLASS_NAMES:
                    prompt_name = CLASS_PROMPT_NAMES[class_name]
                    texts = [template.format(prompt_name) for template in PROMPTS]
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        encoded = adapter["encode_texts"](texts)
                    encoded = F.normalize(encoded.float(), dim=-1)
                    vectors.append(F.normalize(encoded.mean(0), dim=0).cpu().numpy())
            return normalize(np.stack(vectors))


        def embedding_cache_path(split_name, spec):
            ids = SPLIT_FRAMES[split_name]["sample_id"].astype(str).tolist()
            payload = {
                "split": split_name,
                "sample_ids_sha256": hashlib.sha256("\0".join(ids).encode()).hexdigest(),
                "manifest_id": MANIFEST_ID,
                "encoder": spec,
                "preprocess_contract": PREPROCESS_CONTRACTS[spec["backend"]],
            }
            key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            return CACHE_DIR / f"uie22k_{split_name}_{spec['key']}_{key}.npz"


        def compatible_embedding_cache(
            split_name, spec, expected_ids, expected_labels, local_path
        ):
            expected_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "spec": spec,
                        "preprocess_contract": PREPROCESS_CONTRACTS[spec["backend"]],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            legacy_fingerprint = hashlib.sha256(
                json.dumps(spec, sort_keys=True).encode()
            ).hexdigest()
            search_directories = [CACHE_DIR]
            kaggle_input = Path("/kaggle/input")
            notebook_inputs = kaggle_input / "notebooks"
            if notebook_inputs.is_dir():
                search_directories.extend(
                    path
                    for path in notebook_inputs.glob(
                        f"*/*/EvidenceMem/cache/{PROTOCOL_ID}"
                    )
                    if path.is_dir()
                )
            if kaggle_input.is_dir():
                search_directories.extend(
                    path / "EvidenceMem" / "cache" / PROTOCOL_ID
                    for path in kaggle_input.iterdir()
                    if (path / "EvidenceMem" / "cache" / PROTOCOL_ID).is_dir()
                )

            pattern = f"uie22k_{split_name}_{spec['key']}_*.npz"
            candidates = [local_path]
            for directory in search_directories:
                candidates.extend(sorted(directory.glob(pattern)))
            seen = set()
            for candidate in candidates:
                resolved = str(candidate.resolve()) if candidate.exists() else str(candidate)
                if resolved in seen or not candidate.is_file():
                    continue
                seen.add(resolved)
                if not candidate.with_suffix(".json").is_file():
                    continue
                try:
                    cached = load_embedding_cache(candidate)
                    manifest = cached.manifest
                    allowed_fingerprints = {expected_fingerprint}
                    if spec["backend"] == "open_clip":
                        # v4 Version 2 used identical OpenCLIP preprocessing but keyed
                        # its cache by the repository commit.
                        allowed_fingerprints.add(legacy_fingerprint)
                    compatible = (
                        manifest.dataset == "UIE-22K"
                        and manifest.split == split_name
                        and manifest.model_name == spec["model"]
                        and manifest.pretrained == spec["weights"]
                        and manifest.preprocess_fingerprint in allowed_fingerprints
                        and np.array_equal(
                            cached.sample_ids.astype(str), expected_ids
                        )
                        and np.array_equal(cached.labels, expected_labels)
                    )
                    if compatible:
                        return candidate, cached
                except Exception as error:
                    print(f"Ignoring invalid embedding cache {candidate}: {error}")
            return None, None


        def encode_split(adapter, spec, split_name):
            frame = SPLIT_FRAMES[split_name]
            path = embedding_cache_path(split_name, spec)
            expected_ids = frame["sample_id"].astype(str).to_numpy()
            expected_labels = frame["label"].map(CLASS_TO_ID).to_numpy(np.int64)
            cache_path, cached = compatible_embedding_cache(
                split_name, spec, expected_ids, expected_labels, path
            )
            if cached is not None:
                return cached.embeddings, cached.labels, {
                    "encoder_key": spec["key"],
                    "backend": spec["backend"],
                    "resolution": int(spec["resolution"]),
                    "split": split_name,
                    "cache_hit": True,
                    "cache_source": str(cache_path),
                    "seconds": 0.0,
                    "images_per_second": None,
                    "peak_vram_gb": None,
                    "batch_size": None,
                }

            dataset = ManifestImageDataset(
                frame, DATASET_ROOT, adapter["preprocess"]
            )
            batch_size = int(spec["batch_size"])
            while True:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=CFG.num_workers,
                    pin_memory=True,
                    persistent_workers=CFG.num_workers > 0,
                )
                embeddings, labels = [], []
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    with torch.inference_mode():
                        for images, batch_labels in tqdm(
                            loader, desc=f"{spec['key']}: {split_name}"
                        ):
                            images = images.to(DEVICE, non_blocking=True)
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                batch_embeddings = adapter["encode_images"](images)
                            embeddings.append(
                                F.normalize(batch_embeddings.float(), dim=-1).cpu().numpy()
                            )
                            labels.append(batch_labels.numpy())
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    break
                except RuntimeError as error:
                    if "out of memory" not in str(error).lower() or batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    print(
                        f"CUDA OOM; retrying {spec['key']} with batch {batch_size}."
                    )
                    del loader, embeddings, labels
                    torch.cuda.empty_cache()

            matrix = normalize(np.concatenate(embeddings))
            label_array = np.concatenate(labels).astype(np.int64)
            preprocess_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "spec": spec,
                        "preprocess_contract": PREPROCESS_CONTRACTS[spec["backend"]],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            save_embedding_cache(
                path,
                matrix.astype(np.float32),
                label_array,
                expected_ids,
                dataset="UIE-22K",
                split=split_name,
                model_name=spec["model"],
                pretrained=spec["weights"],
                preprocess_fingerprint=preprocess_fingerprint,
                source_revision=git_revision(REPO_DIR),
            )
            runtime = {
                "encoder_key": spec["key"],
                "backend": spec["backend"],
                "resolution": int(spec["resolution"]),
                "split": split_name,
                "cache_hit": False,
                "cache_source": str(path),
                "seconds": float(elapsed),
                "images_per_second": float(len(dataset) / max(elapsed, 1e-12)),
                "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 2**30),
                "batch_size": int(batch_size),
            }
            return matrix, label_array, runtime


        ENCODER_DATA = {}
        encoder_runtime_rows = []
        for encoder_key, spec in ENCODER_SPECS.items():
            seed_everything(CFG.sample_seed)
            adapter = load_encoder(spec)
            text_prototypes = encode_text_prototypes(adapter)
            split_data = {}
            for split_name in ("train", "val", EVALUATION_SPLIT):
                matrix, labels, runtime = encode_split(adapter, spec, split_name)
                split_data[split_name] = (matrix, labels)
                encoder_runtime_rows.append(runtime)
                atomic_csv(
                    pd.DataFrame(encoder_runtime_rows),
                    RUN_DIR / "encoder_runtime.csv",
                )
                journal("embedding_split_ready", **runtime)
            split_data["text_prototypes"] = text_prototypes
            split_data["spec"] = dict(spec)
            ENCODER_DATA[encoder_key] = split_data
            del adapter
            torch.cuda.empty_cache()

        encoder_runtime_df = pd.DataFrame(encoder_runtime_rows)
        atomic_csv(encoder_runtime_df, RUN_DIR / "encoder_runtime.csv")
        for encoder_key, data in ENCODER_DATA.items():
            train_x, train_y = data["train"]
            val_x, val_y = data["val"]
            evaluation_x, evaluation_y = data[EVALUATION_SPLIT]
            assert train_x.shape[0] == CFG.train_per_class * N_CLASSES
            assert val_x.shape[0] == CFG.val_per_class * N_CLASSES
            expected_evaluation = (
                CFG.development_per_class
                if EVALUATION_SPLIT == "development"
                else CFG.confirmatory_per_class
            ) * N_CLASSES
            assert evaluation_x.shape[0] == expected_evaluation
            assert np.allclose(np.linalg.norm(train_x[:100], axis=1), 1.0, atol=2e-3)
            print(
                encoder_key,
                train_x.shape,
                val_x.shape,
                evaluation_x.shape,
            )
        journal("embeddings_ready", runtimes=encoder_runtime_rows)
        display(encoder_runtime_df)
        ''',
        "embeddings",
    ),
    markdown(
        r'''
        ## 3. EvidenceMem v4 and matched baselines

        EvidenceMem v4 starts from the exact same facility-selected real-image medoids
        as the matched coverage baseline. It maps cosine compactness and text alignment
        from `[-1, 1]` to `[0, 1]`, keeps purity on `[0, 1]`, excludes missing alignment
        from the active denominator, and tunes component weights on validation data.
        The scorer compares an equal top-k set inside every class using log-mean-exp.
        Fixed and continuous reliability-conditioned visual-text fusion are reported
        separately, together with a global-scoring ablation.
        '''
    ),
    code(
        r'''
        METHOD_MAP = {"medoid": "kmeans_medoids"}
        SELECTION_CONFIG = SelectionConfig(
            candidate_multiplier=2.0,
            coverage_weight=0.75,
            reliability_weight=0.25,
        )


        def arrays_to_memory(arrays):
            return {
                "prototypes": arrays.prototypes,
                "labels": arrays.labels,
                "source_idx": arrays.source_indices,
                "compactness": arrays.compactness,
                "purity": arrays.purity,
                "alignment": arrays.text_alignment,
                "reliability": arrays.reliabilities,
                "method": arrays.method,
                "budget": arrays.budget_per_class,
                "seed": arrays.seed,
            }


        def memory_to_arrays(memory):
            return PrototypeArrays(
                prototypes=np.asarray(memory["prototypes"], np.float32),
                labels=np.asarray(memory["labels"], np.int64),
                source_indices=np.asarray(memory["source_idx"], np.int64),
                reliabilities=np.asarray(memory["reliability"], np.float32),
                compactness=np.asarray(memory["compactness"], np.float32),
                purity=np.asarray(memory["purity"], np.float32),
                text_alignment=np.asarray(memory["alignment"], np.float32),
                method=str(memory.get("method", "notebook")),
                budget_per_class=int(memory.get("budget", 1)),
                seed=int(memory.get("seed", 0)),
            )


        def fit_memory(x, y, budget, method, seed, text_prototypes):
            arrays = fit_prototype_memory(
                normalize(x),
                np.asarray(y, dtype=np.int64),
                budget_per_class=int(budget),
                method=METHOD_MAP.get(method, method),
                seed=int(seed),
                text_prototypes=normalize(text_prototypes),
                selection_config=SELECTION_CONFIG,
                purity_k=32,
            )
            return arrays_to_memory(arrays)


        def memory_file(encoder_key, method, budget, seed, count):
            selection_id = hashlib.sha256(str(SELECTION_CONFIG).encode()).hexdigest()[:8]
            return CACHE_DIR / (
                f"memory_uie_v4_{MANIFEST_ID[:10]}_{SOURCE_ID[:10]}_"
                f"{encoder_key}_{method}_"
                f"b{budget}_s{seed}_n{count}_{selection_id}.npz"
            )


        def fit_or_load_memory(
            x, y, budget, method, seed, encoder_key, text_prototypes
        ):
            path = memory_file(encoder_key, method, budget, seed, len(x))
            if path.is_file():
                return arrays_to_memory(PrototypeArrays.load(path))
            memory = fit_memory(x, y, budget, method, seed, text_prototypes)
            memory_to_arrays(memory).save(path)
            return memory


        def reliability_weights(values):
            compactness, purity, alignment = (float(value) for value in values)
            return ReliabilityWeights(
                compactness=compactness,
                purity=purity,
                text_alignment=alignment,
            )


        def fit_v4_memory(
            x,
            y,
            budget,
            seed,
            encoder_key,
            text_prototypes,
            weight_values,
        ):
            coverage = fit_or_load_memory(
                x,
                y,
                budget,
                "facility_no_reliability",
                seed,
                encoder_key,
                text_prototypes,
            )
            arrays = reweight_prototype_reliability(
                memory_to_arrays(coverage),
                reliability_weights(weight_values),
                minimum=0.05,
            )
            result = arrays_to_memory(arrays)
            result["method"] = "evidencemem_v4"
            return result


        def memory_scores(memory, queries, alpha, k, text_prototypes):
            fused, visual, text = fused_class_scores(
                memory_to_arrays(memory),
                queries,
                normalize(text_prototypes),
                text_weight=float(alpha),
                k=int(k),
            )
            safe_log = lambda scores: np.log(np.clip(scores, 1e-12, None))
            return safe_log(fused), safe_log(visual), safe_log(text)


        def v4_components(
            memory,
            queries,
            class_k,
            reliability_power,
            text_prototypes,
        ):
            arrays = memory_to_arrays(memory)
            visual, selected_by_class = class_conditional_visual_scores(
                arrays,
                queries,
                n_classes=N_CLASSES,
                k_per_class=int(class_k),
                temperature=0.07,
                reliability_power=float(reliability_power),
            )
            text = text_class_scores(queries, normalize(text_prototypes))

            valid = selected_by_class >= 0
            safe_indices = np.clip(selected_by_class, 0, len(arrays.reliabilities) - 1)
            selected_reliability = arrays.reliabilities[safe_indices] * valid
            class_reliability = selected_reliability.sum(2) / np.clip(
                valid.sum(2), 1, None
            )
            query_reliability = np.sum(visual * class_reliability, axis=1)
            return (
                visual.astype(np.float32),
                text.astype(np.float32),
                query_reliability.astype(np.float32),
                selected_by_class,
            )


        def v4_score_variants(memory, queries, setting, text_prototypes):
            visual, text, query_reliability, selected_by_class = v4_components(
                memory,
                queries,
                setting["class_k"],
                setting["reliability_power"],
                text_prototypes,
            )
            reliability_z = (
                query_reliability - setting["reliability_mean"]
            ) / max(setting["reliability_std"], 1e-6)
            alpha_fixed = np.full(
                len(queries), setting["fixed_alpha"], dtype=np.float32
            )
            alpha_continuous = np.clip(
                setting["continuous_intercept"]
                + setting["continuous_slope"] * reliability_z,
                0.0,
                1.0,
            ).astype(np.float32)

            safe_log = lambda scores: np.log(np.clip(scores, 1e-12, None))
            variants = {}
            for name, alpha in {
                "visual": np.zeros(len(queries), dtype=np.float32),
                "fixed": alpha_fixed,
                "continuous": alpha_continuous,
            }.items():
                fused = (1.0 - alpha[:, None]) * visual + alpha[:, None] * text
                prediction = fused.argmax(1)
                selected = selected_by_class[np.arange(len(prediction)), prediction]
                variants[name] = {
                    "scores": safe_log(fused),
                    "visual_scores": safe_log(visual),
                    "text_scores": safe_log(text),
                    "query_reliability": query_reliability,
                    "selected": selected,
                    "text_weight": alpha,
                }
            return variants


        def tune_memory(memory, validation_x, validation_y, text_prototypes):
            candidates = []
            for k in CFG.topk_grid:
                for alpha in CFG.alpha_grid:
                    scores = memory_scores(
                        memory, validation_x, alpha, k, text_prototypes
                    )[0]
                    candidates.append(
                        (
                            accuracy_score(validation_y, scores.argmax(1)),
                            -int(k),
                            -float(alpha),
                            {"alpha": float(alpha), "k": int(k)},
                        )
                    )
            return max(candidates, key=lambda item: item[:-1])[-1]


        def tune_v4_memory(
            train_x,
            train_y,
            budget,
            seed,
            encoder_key,
            validation_x,
            validation_y,
            text_prototypes,
        ):
            candidates = []
            tuning_records = []
            for weight_values in CFG.reliability_weight_grid:
                memory = fit_v4_memory(
                    train_x,
                    train_y,
                    budget,
                    seed,
                    encoder_key,
                    text_prototypes,
                    weight_values,
                )
                for class_k in CFG.class_topk_grid:
                    for reliability_power in CFG.reliability_power_grid:
                        visual, text, query_reliability, _ = v4_components(
                            memory,
                            validation_x,
                            class_k,
                            reliability_power,
                            text_prototypes,
                        )
                        accuracy = accuracy_score(validation_y, visual.argmax(1))
                        nll = log_loss(
                            validation_y,
                            np.clip(visual, 1e-12, 1.0),
                            labels=np.arange(N_CLASSES),
                        )
                        tuning_records.append(
                            {
                                "weights_compactness": weight_values[0],
                                "weights_purity": weight_values[1],
                                "weights_alignment": weight_values[2],
                                "class_k": int(class_k),
                                "reliability_power": float(reliability_power),
                                "validation_accuracy": float(accuracy),
                                "validation_nll": float(nll),
                            }
                        )
                        candidates.append(
                            (
                                float(accuracy),
                                -float(nll),
                                -int(class_k),
                                -float(reliability_power),
                                weight_values,
                                int(class_k),
                                float(reliability_power),
                                memory,
                                visual,
                                text,
                                query_reliability,
                            )
                        )

            (
                _,
                _,
                _,
                _,
                best_weights,
                best_class_k,
                best_power,
                best_memory,
                best_visual,
                best_text,
                best_query_reliability,
            ) = max(candidates, key=lambda item: item[:4])

            fixed_candidates = []
            for alpha in CFG.alpha_grid:
                fused = (1.0 - alpha) * best_visual + alpha * best_text
                fixed_candidates.append(
                    (
                        accuracy_score(validation_y, fused.argmax(1)),
                        -log_loss(
                            validation_y,
                            np.clip(fused, 1e-12, 1.0),
                            labels=np.arange(N_CLASSES),
                        ),
                        -float(alpha),
                        float(alpha),
                    )
                )
            fixed_alpha = max(fixed_candidates, key=lambda item: item[:3])[-1]

            reliability_mean = float(np.mean(best_query_reliability))
            reliability_std = float(np.std(best_query_reliability))
            reliability_z = (best_query_reliability - reliability_mean) / max(
                reliability_std, 1e-6
            )
            continuous_candidates = []
            for intercept in CFG.alpha_grid:
                for slope in CFG.gate_slope_grid:
                    alpha = np.clip(intercept + slope * reliability_z, 0.0, 1.0)
                    fused = (1.0 - alpha[:, None]) * best_visual + alpha[:, None] * best_text
                    continuous_candidates.append(
                        (
                            accuracy_score(validation_y, fused.argmax(1)),
                            -log_loss(
                                validation_y,
                                np.clip(fused, 1e-12, 1.0),
                                labels=np.arange(N_CLASSES),
                            ),
                            -abs(float(slope)),
                            -float(intercept),
                            float(intercept),
                            float(slope),
                        )
                    )
            continuous_best = max(
                continuous_candidates, key=lambda item: item[:4]
            )
            setting = {
                "variant": "class_conditional_continuous_fusion",
                "reliability_weights": {
                    "compactness": float(best_weights[0]),
                    "purity": float(best_weights[1]),
                    "alignment": float(best_weights[2]),
                },
                "class_k": int(best_class_k),
                "reliability_power": float(best_power),
                "reliability_mean": reliability_mean,
                "reliability_std": reliability_std,
                "fixed_alpha": float(fixed_alpha),
                "continuous_intercept": continuous_best[-2],
                "continuous_slope": continuous_best[-1],
            }
            setting["global_setting"] = tune_memory(
                best_memory, validation_x, validation_y, text_prototypes
            )
            return setting, best_memory, tuning_records


        def full_knn_scores(train_x, train_y, queries, k):
            probabilities = weighted_knn_scores(
                train_x,
                train_y,
                queries,
                k=int(k),
                n_classes=N_CLASSES,
            )
            return np.log(np.clip(probabilities, 1e-12, None))


        def nearest_memory_indices(memory, queries, k):
            _, selected = exact_search(memory["prototypes"], queries, int(k))
            return selected


        def evidence_precision_at_k(memory, selected, query_labels):
            labels = np.asarray(memory["labels"], np.int64)[np.asarray(selected)]
            return float(np.mean(labels == np.asarray(query_labels)[:, None]))
        ''',
        "methods",
    ),
    code(
        r'''
        # Metrics, validation-only calibration, paired tests, and multiple-testing correction.
        def softmax_np(scores, temperature):
            logits = np.asarray(scores, np.float64) / float(temperature)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            return probabilities / probabilities.sum(axis=1, keepdims=True)


        def expected_calibration_error(probabilities, labels, bins=15):
            confidence = probabilities.max(1)
            prediction = probabilities.argmax(1)
            edges = np.linspace(0.0, 1.0, bins + 1)
            value = 0.0
            for lower, upper in zip(edges[:-1], edges[1:], strict=False):
                mask = (confidence > lower) & (confidence <= upper)
                if np.any(mask):
                    value += mask.mean() * abs(
                        (prediction[mask] == labels[mask]).mean()
                        - confidence[mask].mean()
                    )
            return float(value)


        def selective_aurc(probabilities, labels):
            confidence = probabilities.max(1)
            prediction = probabilities.argmax(1)
            errors = (prediction != labels).astype(np.float64)
            order = np.argsort(-confidence, kind="stable")
            risks = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
            coverage = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
            if len(errors) == 1:
                return float(risks[0])
            widths = np.diff(coverage)
            return float(np.sum((risks[:-1] + risks[1:]) * 0.5 * widths))


        def select_temperature(validation_scores, labels):
            losses = {
                temperature: log_loss(
                    labels,
                    softmax_np(validation_scores, temperature),
                    labels=np.arange(validation_scores.shape[1]),
                )
                for temperature in CFG.temperatures
            }
            return min(losses, key=losses.get)


        def select_confidence_threshold(
            validation_scores, temperature, target_coverage=0.90
        ):
            validation_probabilities = softmax_np(validation_scores, temperature)
            confidence = validation_probabilities.max(1)
            return float(np.quantile(confidence, 1.0 - float(target_coverage)))


        def evaluate_scores(
            name,
            evaluation_scores,
            labels,
            temperature,
            confidence_threshold,
            seed,
            encoder_key,
            resolution,
            extra=None,
        ):
            uncalibrated_probabilities = softmax_np(evaluation_scores, 1.0)
            probabilities = softmax_np(evaluation_scores, temperature)
            prediction = probabilities.argmax(1)
            one_hot = np.eye(evaluation_scores.shape[1])[labels]
            retained = probabilities.max(1) >= float(confidence_threshold)
            row = {
                "method": name,
                "seed": int(seed),
                "encoder_key": str(encoder_key),
                "resolution": int(resolution),
                "evaluation_stage": EVALUATION_STAGE,
                "evaluation_split": EVALUATION_SPLIT,
                "n": int(len(labels)),
                "accuracy": float(accuracy_score(labels, prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
                "macro_f1": float(f1_score(labels, prediction, average="macro")),
                "nll": float(
                    log_loss(
                        labels,
                        probabilities,
                        labels=np.arange(evaluation_scores.shape[1]),
                    )
                ),
                "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
                "ece_15": expected_calibration_error(probabilities, labels),
                "ece_15_uncalibrated": expected_calibration_error(
                    uncalibrated_probabilities, labels
                ),
                "nll_uncalibrated": float(
                    log_loss(
                        labels,
                        uncalibrated_probabilities,
                        labels=np.arange(evaluation_scores.shape[1]),
                    )
                ),
                "aurc": selective_aurc(probabilities, labels),
                "temperature": float(temperature),
                "confidence_threshold_90pct_validation_coverage": float(
                    confidence_threshold
                ),
                "selective_coverage": float(retained.mean()),
                "selective_accuracy": float(
                    accuracy_score(labels[retained], prediction[retained])
                    if np.any(retained)
                    else np.nan
                ),
            }
            if extra:
                row.update(extra)
            return row, prediction, probabilities


        def paired_bootstrap(correct_a, correct_b, seed=2026, draws=4_000):
            differences = np.asarray(correct_a, float) - np.asarray(correct_b, float)
            rng = np.random.default_rng(seed)
            samples = np.empty(draws)
            for start in range(0, draws, 200):
                stop = min(start + 200, draws)
                ids = rng.integers(
                    0, len(differences), size=(stop - start, len(differences))
                )
                samples[start:stop] = differences[ids].mean(1)
            lower_tail = (np.count_nonzero(samples <= 0) + 1) / (draws + 1)
            upper_tail = (np.count_nonzero(samples >= 0) + 1) / (draws + 1)
            return {
                "delta": float(differences.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "p_two_sided": float(min(1.0, 2 * min(lower_tail, upper_tail))),
                "draws": int(draws),
            }


        def mcnemar_exact(correct_a, correct_b):
            from scipy.stats import binomtest

            correct_a = np.asarray(correct_a, dtype=bool)
            correct_b = np.asarray(correct_b, dtype=bool)
            a_only = int(np.sum(correct_a & ~correct_b))
            b_only = int(np.sum(~correct_a & correct_b))
            p_value = (
                1.0
                if a_only + b_only == 0
                else binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue
            )
            return {
                "a_only_correct": a_only,
                "b_only_correct": b_only,
                "p_exact": float(p_value),
            }


        def benjamini_hochberg(p_values):
            values = np.asarray(p_values, dtype=float)
            order = np.argsort(values)
            adjusted = np.empty_like(values)
            running = 1.0
            for rank_index in range(len(values) - 1, -1, -1):
                original_index = order[rank_index]
                running = min(
                    running,
                    values[original_index] * len(values) / (rank_index + 1),
                )
                adjusted[original_index] = running
            return np.clip(adjusted, 0.0, 1.0)


        def timed_scores(function):
            started = time.perf_counter()
            result = function()
            elapsed = time.perf_counter() - started
            return result, elapsed


        def safe_filename(value):
            return "".join(
                character if character.isalnum() else "_" for character in value
            ).strip("_")
        ''',
        "metrics",
    ),
    markdown(
        r'''
        ## 4. Main comparison

        Each native encoder is evaluated on the same samples. The comparison includes
        zero-shot vision-text scoring, full-training-set weighted kNN, a linear probe,
        matched prototype-memory baselines, Tip-Adapter, and four EvidenceMem v4
        scoring variants. Hyperparameters, temperatures, and the 90% selective-risk
        threshold are chosen on validation only. Query-level predictions support
        paired bootstrap intervals and exact McNemar tests.
        '''
    ),
    code(
        r'''
        classification_rows = []
        tuning_rows = []
        prediction_bank = {}
        primary_memories = {}
        qualitative_bank = {}
        selected_hyperparameters = {}

        for resolution in CFG.resolutions:
            data = RESOLUTION_DATA[int(resolution)]
            train_x, train_y = data["train"]
            val_x, val_y = data["val"]
            test_x, test_y = data["test"]
            text_prototypes = data["text_prototypes"]

            knn_candidates = []
            for k in CFG.topk_grid:
                validation_scores = full_knn_scores(train_x, train_y, val_x, k)
                knn_candidates.append(
                    (accuracy_score(val_y, validation_scores.argmax(1)), -int(k), int(k))
                )
            selected_knn_k = max(knn_candidates)[-1]
            knn_val = full_knn_scores(train_x, train_y, val_x, selected_knn_k)
            (knn_test, knn_seconds) = timed_scores(
                lambda: full_knn_scores(train_x, train_y, test_x, selected_knn_k)
            )

            probe_candidates = []
            for c_value in (0.01, 0.1, 1.0, 10.0):
                probe = LogisticRegression(
                    C=c_value,
                    max_iter=2_000,
                    solver="lbfgs",
                    random_state=CFG.seeds[0],
                )
                probe.fit(train_x, train_y)
                validation_scores = probe.decision_function(val_x)
                probe_candidates.append(
                    (
                        accuracy_score(val_y, validation_scores.argmax(1)),
                        -float(c_value),
                        float(c_value),
                        probe,
                        validation_scores,
                    )
                )
            _, _, selected_probe_c, selected_probe, probe_val = max(
                probe_candidates, key=lambda item: item[:2]
            )
            (probe_test, probe_seconds) = timed_scores(
                lambda: selected_probe.decision_function(test_x)
            )

            for seed in CFG.seeds:
                seed_everything(seed)
                memories = {
                    "Random memory": fit_or_load_memory(
                        train_x,
                        train_y,
                        CFG.default_budget,
                        "random",
                        seed,
                        resolution,
                        text_prototypes,
                    ),
                    "Centroid": fit_or_load_memory(
                        train_x,
                        train_y,
                        1,
                        "centroid",
                        seed,
                        resolution,
                        text_prototypes,
                    ),
                    "KMeans medoids": fit_or_load_memory(
                        train_x,
                        train_y,
                        CFG.default_budget,
                        "medoid",
                        seed,
                        resolution,
                        text_prototypes,
                    ),
                    "Facility selection (no reliability)": fit_or_load_memory(
                        train_x,
                        train_y,
                        CFG.default_budget,
                        "facility_no_reliability",
                        seed,
                        resolution,
                        text_prototypes,
                    ),
                    "EvidenceMem v2 selection": fit_or_load_memory(
                        train_x,
                        train_y,
                        CFG.default_budget,
                        "evidencemem",
                        seed,
                        resolution,
                        text_prototypes,
                    ),
                }
                primary_memory = fit_v3_memory(
                    train_x,
                    train_y,
                    CFG.default_budget,
                    seed,
                    resolution,
                    text_prototypes,
                )
                primary_memories[(int(resolution), int(seed))] = primary_memory

                standard_settings = {}
                for memory_name, memory in memories.items():
                    setting = tune_memory(memory, val_x, val_y, text_prototypes)
                    standard_settings[memory_name] = setting
                    for k in CFG.topk_grid:
                        for alpha in CFG.alpha_grid:
                            scores = memory_scores(
                                memory, val_x, alpha, k, text_prototypes
                            )[0]
                            tuning_rows.append(
                                {
                                    "resolution": int(resolution),
                                    "seed": int(seed),
                                    "method": memory_name,
                                    "alpha": float(alpha),
                                    "k": int(k),
                                    "validation_accuracy": float(
                                        accuracy_score(val_y, scores.argmax(1))
                                    ),
                                }
                            )
                v3_setting = tune_v3_memory(
                    primary_memory, val_x, val_y, text_prototypes
                )
                selected_hyperparameters[f"r{resolution}_s{seed}"] = {
                    "standard_methods": standard_settings,
                    "evidencemem_v3": v3_setting,
                    "full_knn_k": selected_knn_k,
                    "linear_probe_c": selected_probe_c,
                }

                score_sets = {
                    "CLIP zero-shot": {
                        "val": val_x @ text_prototypes.T,
                        "test": test_x @ text_prototypes.T,
                        "seconds": 0.0,
                        "setting": {},
                        "memory": None,
                        "selected": None,
                    },
                    "Full kNN": {
                        "val": knn_val,
                        "test": knn_test,
                        "seconds": knn_seconds,
                        "setting": {"k": selected_knn_k},
                        "memory": None,
                        "selected": None,
                    },
                    "Linear probe": {
                        "val": probe_val,
                        "test": probe_test,
                        "seconds": probe_seconds,
                        "setting": {"c": selected_probe_c},
                        "memory": None,
                        "selected": None,
                    },
                }

                for memory_name, memory in memories.items():
                    setting = standard_settings[memory_name]
                    validation_scores = memory_scores(
                        memory,
                        val_x,
                        setting["alpha"],
                        setting["k"],
                        text_prototypes,
                    )[0]
                    (test_scores, elapsed) = timed_scores(
                        lambda memory=memory, setting=setting: memory_scores(
                            memory,
                            test_x,
                            setting["alpha"],
                            setting["k"],
                            text_prototypes,
                        )[0]
                    )
                    selected = nearest_memory_indices(memory, test_x, setting["k"])
                    score_sets[f"{memory_name} fused"] = {
                        "val": validation_scores,
                        "test": test_scores,
                        "seconds": elapsed,
                        "setting": setting,
                        "memory": memory,
                        "selected": selected,
                    }

                v3_val = v3_memory_scores(
                    primary_memory, val_x, v3_setting, text_prototypes
                )
                (v3_test, v3_seconds) = timed_scores(
                    lambda: v3_memory_scores(
                        primary_memory, test_x, v3_setting, text_prototypes
                    )
                )
                for method_name, validation_scores, test_scores in (
                    ("EvidenceMem v3 visual", v3_val[1], v3_test[1]),
                    ("EvidenceMem v3 gated", v3_val[0], v3_test[0]),
                ):
                    score_sets[method_name] = {
                        "val": validation_scores,
                        "test": test_scores,
                        "seconds": v3_seconds,
                        "setting": v3_setting,
                        "memory": primary_memory,
                        "selected": v3_test[4],
                        "query_reliability": v3_test[3],
                    }

                tip_cache = memory_to_arrays(memories["Random memory"])
                tip_candidates = []
                for beta in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
                    for cache_weight in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
                        scores = tip_adapter_scores(
                            tip_cache,
                            val_x,
                            text_prototypes,
                            beta=beta,
                            cache_weight=cache_weight,
                        )
                        tip_candidates.append(
                            (
                                accuracy_score(val_y, scores.argmax(1)),
                                -beta,
                                -cache_weight,
                                beta,
                                cache_weight,
                                scores,
                            )
                        )
                _, _, _, tip_beta, tip_weight, tip_val = max(
                    tip_candidates, key=lambda item: item[:3]
                )
                (tip_test, tip_seconds) = timed_scores(
                    lambda: tip_adapter_scores(
                        tip_cache,
                        test_x,
                        text_prototypes,
                        beta=tip_beta,
                        cache_weight=tip_weight,
                    )
                )
                tip_selected = nearest_memory_indices(
                    memories["Random memory"], test_x, min(10, len(tip_cache.labels))
                )
                score_sets["Tip-Adapter (matched cache)"] = {
                    "val": tip_val,
                    "test": tip_test,
                    "seconds": tip_seconds,
                    "setting": {"beta": tip_beta, "cache_weight": tip_weight},
                    "memory": memories["Random memory"],
                    "selected": tip_selected,
                }

                for method_name, packet in score_sets.items():
                    temperature = select_temperature(packet["val"], val_y)
                    setting = packet["setting"]
                    evidence_precision = np.nan
                    if packet["memory"] is not None and packet["selected"] is not None:
                        evidence_precision = evidence_precision_at_k(
                            packet["memory"], packet["selected"], test_y
                        )
                    extra = {
                        "stored_examples": (
                            len(packet["memory"]["labels"])
                            if packet["memory"] is not None
                            else (len(train_y) if method_name == "Full kNN" else 0)
                        ),
                        "budget_per_class": (
                            int(packet["memory"]["budget"])
                            if packet["memory"] is not None
                            else np.nan
                        ),
                        "selected_alpha": setting.get("alpha", np.nan),
                        "selected_k": setting.get("k", np.nan),
                        "selected_gamma": setting.get("gamma", np.nan),
                        "selected_gate_threshold": setting.get(
                            "gate_threshold", np.nan
                        ),
                        "selected_alpha_reliable": setting.get(
                            "alpha_reliable", np.nan
                        ),
                        "selected_alpha_uncertain": setting.get(
                            "alpha_uncertain", np.nan
                        ),
                        "evidence_precision_at_k": evidence_precision,
                        "query_reliability_mean": float(
                            np.mean(packet.get("query_reliability", np.array([np.nan])))
                        ),
                        "inference_ms_per_query": float(
                            1_000.0 * packet["seconds"] / len(test_y)
                        ),
                    }
                    row, prediction, probabilities = evaluate_scores(
                        method_name,
                        packet["test"],
                        test_y,
                        temperature,
                        seed,
                        resolution,
                        extra,
                    )
                    classification_rows.append(row)
                    prediction_bank[(int(resolution), int(seed), method_name)] = prediction
                    prediction_path = RUN_DIR / (
                        f"predictions_r{resolution}_s{seed}_{safe_filename(method_name)}.npz"
                    )
                    np.savez_compressed(
                        prediction_path,
                        labels=test_y,
                        predictions=prediction,
                        probabilities=probabilities,
                        scores=packet["test"],
                    )

                qualitative_bank[(int(resolution), int(seed))] = {
                    "prediction": prediction_bank[
                        (int(resolution), int(seed), "EvidenceMem v3 gated")
                    ],
                    "probabilities": softmax_np(
                        v3_test[0],
                        select_temperature(v3_val[0], val_y),
                    ),
                    "query_reliability": v3_test[3],
                    "selected": v3_test[4],
                    "labels": test_y,
                }
                atomic_csv(
                    pd.DataFrame(classification_rows),
                    RUN_DIR / "classification_results.csv",
                )
                atomic_csv(
                    pd.DataFrame(tuning_rows),
                    RUN_DIR / "fusion_topk_validation.csv",
                )
                atomic_json(
                    RUN_DIR / "selected_hyperparameters.json",
                    selected_hyperparameters,
                )
                journal(
                    "classification_seed_complete",
                    resolution=int(resolution),
                    seed=int(seed),
                )

        classification_df = pd.DataFrame(classification_rows)
        classification_summary_df = (
            classification_df.groupby(["resolution", "method"], as_index=False)
            .agg(
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                balanced_accuracy_mean=("balanced_accuracy", "mean"),
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                evidence_precision_mean=("evidence_precision_at_k", "mean"),
                aurc_mean=("aurc", "mean"),
                ece_mean=("ece_15", "mean"),
                inference_ms_per_query_mean=("inference_ms_per_query", "mean"),
            )
            .fillna(0.0)
            .sort_values(["resolution", "accuracy_mean"], ascending=[True, False])
        )
        atomic_csv(
            classification_summary_df, RUN_DIR / "classification_summary.csv"
        )

        paired_results = []
        for resolution in CFG.resolutions:
            labels = RESOLUTION_DATA[int(resolution)]["test"][1]
            for seed in CFG.seeds:
                evidence_correct = (
                    prediction_bank[
                        (int(resolution), int(seed), "EvidenceMem v3 gated")
                    ]
                    == labels
                )
                for baseline in (
                    "Facility selection (no reliability) fused",
                    "KMeans medoids fused",
                    "Tip-Adapter (matched cache)",
                    "Linear probe",
                ):
                    baseline_correct = (
                        prediction_bank[(int(resolution), int(seed), baseline)] == labels
                    )
                    paired_results.append(
                        {
                            "comparison_type": "method",
                            "resolution": int(resolution),
                            "seed": int(seed),
                            "comparison": f"EvidenceMem v3 gated vs {baseline}",
                            "bootstrap": paired_bootstrap(
                                evidence_correct,
                                baseline_correct,
                                seed=CFG.sample_seed + int(seed) + int(resolution),
                            ),
                            "mcnemar": mcnemar_exact(
                                evidence_correct, baseline_correct
                            ),
                        }
                    )

        if 224 in CFG.resolutions and 512 in CFG.resolutions:
            labels = RESOLUTION_DATA[224]["test"][1]
            if not np.array_equal(labels, RESOLUTION_DATA[512]["test"][1]):
                raise AssertionError("Resolution conditions do not share test labels.")
            for seed in CFG.seeds:
                correct_512 = (
                    prediction_bank[(512, int(seed), "EvidenceMem v3 gated")] == labels
                )
                correct_224 = (
                    prediction_bank[(224, int(seed), "EvidenceMem v3 gated")] == labels
                )
                paired_results.append(
                    {
                        "comparison_type": "resolution",
                        "resolution": "512_vs_224",
                        "seed": int(seed),
                        "comparison": "EvidenceMem v3 gated at 512 vs 224",
                        "bootstrap": paired_bootstrap(
                            correct_512,
                            correct_224,
                            seed=CFG.sample_seed + int(seed),
                        ),
                        "mcnemar": mcnemar_exact(correct_512, correct_224),
                    }
                )

        adjusted = benjamini_hochberg(
            [row["mcnemar"]["p_exact"] for row in paired_results]
        )
        for row, adjusted_p in zip(paired_results, adjusted, strict=True):
            row["mcnemar"]["p_bh"] = float(adjusted_p)
        atomic_json(RUN_DIR / "paired_tests.json", paired_results)

        if 224 in CFG.resolutions and 512 in CFG.resolutions:
            paired_frame = classification_df.pivot_table(
                index=["seed", "method"],
                columns="resolution",
                values=["accuracy", "macro_f1", "evidence_precision_at_k", "aurc"],
            ).reset_index()
            paired_frame.columns = [
                "_".join(str(part) for part in column if str(part) != "")
                if isinstance(column, tuple)
                else str(column)
                for column in paired_frame.columns
            ]
            for metric in ("accuracy", "macro_f1", "evidence_precision_at_k", "aurc"):
                paired_frame[f"{metric}_delta_512_minus_224"] = (
                    paired_frame[f"{metric}_512"] - paired_frame[f"{metric}_224"]
                )
            atomic_csv(paired_frame, RUN_DIR / "resolution_deltas.csv")
        else:
            paired_frame = pd.DataFrame()
            atomic_csv(paired_frame, RUN_DIR / "resolution_deltas.csv")

        display(classification_summary_df)
        if not paired_frame.empty:
            display(
                paired_frame.groupby("method", as_index=False)[
                    "accuracy_delta_512_minus_224"
                ].agg(["mean", "std"])
            )
        ''',
        "main-experiment",
    ),
    markdown(
        r'''
        ## 5. Equal-count memory-budget curves

        On the development-selected encoder, random memory, coverage-only facility
        selection, and EvidenceMem v4 continuous fusion are evaluated at 20, 40, and
        80 stored images per class. A hard assertion enforces equal storage before a
        result is accepted.
        '''
    ),
    code(
        r'''
        budget_rows = []
        for resolution in CFG.resolutions:
            data = RESOLUTION_DATA[int(resolution)]
            train_x, train_y = data["train"]
            val_x, val_y = data["val"]
            test_x, test_y = data["test"]
            text_prototypes = data["text_prototypes"]
            for seed in CFG.seeds:
                for budget in CFG.budgets:
                    random_memory = fit_or_load_memory(
                        train_x,
                        train_y,
                        budget,
                        "random",
                        seed,
                        resolution,
                        text_prototypes,
                    )
                    facility_memory = fit_or_load_memory(
                        train_x,
                        train_y,
                        budget,
                        "facility_no_reliability",
                        seed,
                        resolution,
                        text_prototypes,
                    )
                    v3_memory = fit_v3_memory(
                        train_x,
                        train_y,
                        budget,
                        seed,
                        resolution,
                        text_prototypes,
                    )
                    for method_name, memory, is_v3 in (
                        ("Random memory", random_memory, False),
                        (
                            "Facility selection (no reliability)",
                            facility_memory,
                            False,
                        ),
                        ("EvidenceMem v3 gated", v3_memory, True),
                    ):
                        if is_v3:
                            setting = tune_v3_memory(
                                memory, val_x, val_y, text_prototypes
                            )
                            packet = v3_memory_scores(
                                memory, test_x, setting, text_prototypes
                            )
                            scores, selected = packet[0], packet[4]
                        else:
                            setting = tune_memory(
                                memory, val_x, val_y, text_prototypes
                            )
                            scores = memory_scores(
                                memory,
                                test_x,
                                setting["alpha"],
                                setting["k"],
                                text_prototypes,
                            )[0]
                            selected = nearest_memory_indices(
                                memory, test_x, setting["k"]
                            )
                        prediction = scores.argmax(1)
                        budget_rows.append(
                            {
                                "resolution": int(resolution),
                                "seed": int(seed),
                                "method": method_name,
                                "budget_per_class": int(budget),
                                "stored_examples": int(len(memory["labels"])),
                                "accuracy": float(accuracy_score(test_y, prediction)),
                                "balanced_accuracy": float(
                                    balanced_accuracy_score(test_y, prediction)
                                ),
                                "macro_f1": float(
                                    f1_score(test_y, prediction, average="macro")
                                ),
                                "evidence_precision_at_k": evidence_precision_at_k(
                                    memory, selected, test_y
                                ),
                                "selected_setting": json.dumps(setting, sort_keys=True),
                            }
                        )
                    atomic_csv(
                        pd.DataFrame(budget_rows),
                        RUN_DIR / "memory_budget_results.csv",
                    )

        budget_df = pd.DataFrame(budget_rows)
        budget_summary_df = (
            budget_df.groupby(
                ["resolution", "method", "budget_per_class"], as_index=False
            )
            .agg(
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                macro_f1_mean=("macro_f1", "mean"),
                evidence_precision_mean=("evidence_precision_at_k", "mean"),
            )
            .fillna(0.0)
        )
        atomic_csv(budget_summary_df, RUN_DIR / "memory_budget_summary.csv")
        display(budget_summary_df)
        ''',
        "budget-curves",
    ),
    markdown(
        r'''
        ## 6. Qualitative retrieved evidence

        For the first predeclared seed on the selected encoder, this cell exports
        difficult correct cases and confident errors with the three strongest stored
        prototypes inside the predicted class. These are explicitly labeled decision
        evidence; the quantitative evidence metric uses global neighbours instead.
        '''
    ),
    code(
        r'''
        qualitative_rows = []
        for resolution in CFG.resolutions:
            seed = int(CFG.seeds[0])
            packet = qualitative_bank[(int(resolution), seed)]
            memory = primary_memories[(int(resolution), seed)]
            labels = packet["labels"]
            prediction = packet["prediction"]
            confidence = packet["probabilities"].max(1)
            correct = prediction == labels
            wrong_indices = np.flatnonzero(~correct)
            correct_indices = np.flatnonzero(correct)
            selected_examples = []
            if len(wrong_indices):
                selected_examples.extend(
                    wrong_indices[np.argsort(-confidence[wrong_indices])[:3]].tolist()
                )
            if len(correct_indices):
                selected_examples.extend(
                    correct_indices[np.argsort(confidence[correct_indices])[:3]].tolist()
                )
            if not selected_examples:
                selected_examples = np.argsort(confidence)[:6].tolist()

            figure, axes = plt.subplots(
                len(selected_examples), 4, figsize=(10, 2.6 * len(selected_examples))
            )
            axes = np.atleast_2d(axes)
            train_frame = SPLIT_FRAMES["train"]
            test_frame = SPLIT_FRAMES["test"]
            for row_number, test_index in enumerate(selected_examples):
                query_row = test_frame.iloc[int(test_index)]
                query_path = DATASET_ROOT / query_row["relative_path"]
                axes[row_number, 0].imshow(Image.open(query_path).convert("RGB"))
                axes[row_number, 0].set_title(
                    f"Query: {CLASS_NAMES[int(labels[test_index])]}\n"
                    f"Pred: {CLASS_NAMES[int(prediction[test_index])]} "
                    f"({confidence[test_index]:.2f})",
                    fontsize=8,
                )
                selected_prototypes = packet["selected"][int(test_index), :3]
                source_indices = np.asarray(memory["source_idx"], dtype=int)[
                    selected_prototypes
                ]
                evidence_labels = np.asarray(memory["labels"], dtype=int)[
                    selected_prototypes
                ]
                record = {
                    "resolution": int(resolution),
                    "seed": seed,
                    "test_index": int(test_index),
                    "query_path": query_row["relative_path"],
                    "true_label": CLASS_NAMES[int(labels[test_index])],
                    "predicted_label": CLASS_NAMES[int(prediction[test_index])],
                    "correct": bool(correct[test_index]),
                    "confidence": float(confidence[test_index]),
                    "query_reliability": float(packet["query_reliability"][test_index]),
                }
                for evidence_rank, (source_index, evidence_label) in enumerate(
                    zip(source_indices, evidence_labels, strict=True), start=1
                ):
                    evidence_row = train_frame.iloc[int(source_index)]
                    evidence_path = DATASET_ROOT / evidence_row["relative_path"]
                    axes[row_number, evidence_rank].imshow(
                        Image.open(evidence_path).convert("RGB")
                    )
                    axes[row_number, evidence_rank].set_title(
                        f"Evidence {evidence_rank}: "
                        f"{CLASS_NAMES[int(evidence_label)]}",
                        fontsize=8,
                    )
                    record[f"evidence_{evidence_rank}_path"] = evidence_row[
                        "relative_path"
                    ]
                    record[f"evidence_{evidence_rank}_label"] = CLASS_NAMES[
                        int(evidence_label)
                    ]
                qualitative_rows.append(record)
            for axis in axes.ravel():
                axis.axis("off")
            figure.tight_layout()
            figure.savefig(
                RUN_DIR / f"qualitative_evidence_r{resolution}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()
        qualitative_df = pd.DataFrame(qualitative_rows)
        atomic_csv(qualitative_df, RUN_DIR / "qualitative_evidence.csv")
        ''',
        "qualitative-evidence",
    ),
    markdown(
        r'''
        ## 7. Paper figures, integrity gate, and export

        The integrity gate checks manifest balance, complete methods and seeds,
        matched budget rows, calibration output, and paired tests. A development run
        may select one encoder but cannot unlock final claims. Only a paper-mode run
        on the untouched confirmatory split can do that.
        '''
    ),
    code(
        r'''
        selected_methods = [
            "CLIP zero-shot",
            "Full kNN",
            "Linear probe",
            "Facility selection (no reliability) fused",
            "Tip-Adapter (matched cache)",
            "EvidenceMem v3 gated",
        ]
        plot_frame = classification_summary_df[
            classification_summary_df["method"].isin(selected_methods)
        ].copy()

        figure, axis = plt.subplots(figsize=(10, 4.5))
        sns.barplot(
            data=plot_frame,
            x="method",
            y="accuracy_mean",
            hue="resolution",
            palette=["#0072B2", "#D55E00"],
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel("Top-1 accuracy")
        axis.tick_params(axis="x", rotation=25)
        axis.legend(title="Input resolution")
        figure.tight_layout()
        figure.savefig(RUN_DIR / "main_accuracy.pdf", bbox_inches="tight")
        plt.show()

        figure, axis = plt.subplots(figsize=(8, 4.5))
        for (resolution, method), group in budget_summary_df.groupby(
            ["resolution", "method"]
        ):
            style = "-" if int(resolution) == 512 else "--"
            axis.plot(
                group["budget_per_class"],
                group["accuracy_mean"],
                marker="o",
                linestyle=style,
                label=f"{method}, r{resolution}",
            )
        axis.set_xlabel("Stored images per class")
        axis.set_ylabel("Top-1 accuracy")
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(RUN_DIR / "memory_budget_accuracy.pdf", bbox_inches="tight")
        plt.show()

        v3_summary = classification_summary_df[
            classification_summary_df["method"] == "EvidenceMem v3 gated"
        ].set_index("resolution")
        result_summary = {
            "accuracy_by_resolution": {
                str(int(index)): float(row["accuracy_mean"])
                for index, row in v3_summary.iterrows()
            },
            "macro_f1_by_resolution": {
                str(int(index)): float(row["macro_f1_mean"])
                for index, row in v3_summary.iterrows()
            },
        }
        if 224 in v3_summary.index and 512 in v3_summary.index:
            result_summary["accuracy_delta_512_minus_224"] = float(
                v3_summary.loc[512, "accuracy_mean"]
                - v3_summary.loc[224, "accuracy_mean"]
            )
            result_summary["higher_resolution_improved_mean_accuracy"] = bool(
                result_summary["accuracy_delta_512_minus_224"] > 0
            )

        expected_classification_rows = (
            len(CFG.resolutions) * len(CFG.seeds) * N_CLASSES
        )
        expected_split_counts = {
            "train": CFG.train_per_class,
            "val": CFG.val_per_class,
            "test": CFG.test_per_class,
        }
        observed_split_counts = (
            manifest_df.groupby(["label", "split"]).size().unstack(fill_value=0)
        )
        manifest_balanced = (
            set(observed_split_counts.index) == set(CLASS_NAMES)
            and set(observed_split_counts.columns) == set(expected_split_counts)
            and all(
                int(observed_split_counts.at[class_name, split_name])
                == expected_count
                for class_name in CLASS_NAMES
                for split_name, expected_count in expected_split_counts.items()
            )
        )
        integrity_checks = {
            "paper_mode": CFG.mode == "paper",
            "manifest_balanced": bool(manifest_balanced),
            "manifest_has_no_exact_duplicates": not bool(
                manifest_df["sha256"].duplicated().any()
            ),
            "classification_rows_complete": len(classification_df)
            == expected_classification_rows,
            "all_seeds_complete": classification_df["seed"].nunique()
            == len(CFG.seeds),
            "all_resolutions_complete": set(classification_df["resolution"])
            == set(CFG.resolutions),
            "paired_tests_present": len(paired_results) > 0,
            "budget_rows_present": len(budget_df) > 0,
        }
        claim_gate = {
            "protocol_id": PROTOCOL_ID,
            "protocol_revision": PROTOCOL_REVISION,
            "manifest_id": MANIFEST_ID,
            "integrity_checks": integrity_checks,
            "ready_for_interpretation": all(integrity_checks.values()),
            "result_summary": result_summary,
            "warning": (
                "The 512 condition interpolates positional embeddings from the "
                "pretrained 224 model and must be described as an inference-resolution "
                "stress test."
            ),
        }
        atomic_json(RUN_DIR / "claim_gate.json", claim_gate)
        if not claim_gate["ready_for_interpretation"]:
            failed = [name for name, passed in integrity_checks.items() if not passed]
            raise RuntimeError(f"Run integrity gate failed: {failed}")

        required_artifacts = [
            "uie22k_manifest.csv",
            "sampling_summary.json",
            "environment.json",
            "encoder_runtime.csv",
            "classification_results.csv",
            "classification_summary.csv",
            "fusion_topk_validation.csv",
            "selected_hyperparameters.json",
            "paired_tests.json",
            "resolution_deltas.csv",
            "memory_budget_results.csv",
            "memory_budget_summary.csv",
            "qualitative_evidence.csv",
            "main_accuracy.pdf",
            "memory_budget_accuracy.pdf",
            "claim_gate.json",
        ]
        completed_manifest = finalize_run_manifest(
            RUN_MANIFEST,
            run_directory=RUN_DIR,
            required_artifacts=required_artifacts,
        )
        atomic_write_json(RUN_DIR / "run_manifest.json", completed_manifest)
        archive_path = Path(
            shutil.make_archive(str(RUN_DIR), "zip", root_dir=RUN_DIR)
        )
        journal("run_complete", archive=str(archive_path), results=result_summary)
        print(json.dumps(claim_gate, indent=2))
        print("Complete artifact directory:", RUN_DIR)
        print("Downloadable archive:", archive_path)
        try:
            from IPython.display import FileLink, display

            display(FileLink(str(archive_path)))
        except Exception:
            pass
        ''',
        "export",
    ),
]


def replace_tagged_code_cell(tag: str, fragment_name: str) -> None:
    """Replace one generated code cell with a syntax-checkable source fragment."""
    matching = [cell for cell in cells if tag in cell.get("metadata", {}).get("tags", [])]
    if len(matching) != 1:
        raise RuntimeError(f"Expected one notebook cell tagged {tag!r}, found {len(matching)}")
    matching[0]["source"] = source(
        (CELL_DIR / fragment_name).read_text(encoding="utf-8")
    )


replace_tagged_code_cell("main-experiment", "uie22k_v4_main.py")
replace_tagged_code_cell("budget-curves", "uie22k_v4_budget.py")
replace_tagged_code_cell("qualitative-evidence", "uie22k_v4_qualitative.py")
replace_tagged_code_cell("export", "uie22k_v4_export.py")


def main() -> None:
    original = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(cells):
        cell["id"] = f"uie22k-{index:02d}"
    metadata = dict(original.get("metadata", {}))
    metadata["accelerator"] = "GPU"
    metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    metadata["language_info"] = {"name": "python", "version": "3.11"}
    notebook = {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_NOTEBOOK} with {len(cells)} cells")


if __name__ == "__main__":
    main()
