"""Runtime dependency installer for Thermal Delamination Detector.

This module makes a best-effort attempt to ensure that the third-party
packages required by the GUI are installed before the main application code
imports them.  It is designed for "no code" users that may double-click the
``main.py`` script without setting up a Python environment manually.

Only the Python standard library is used here so that we can run even when all
other dependencies are absent.  When a required package is missing we invoke
``python -m pip`` with the current interpreter to install it into the active
environment.  Any installation errors are surfaced to the caller so that the
main application can still present a helpful error dialog.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class Dependency:
    """Metadata that maps import names to the corresponding pip package."""

    import_name: str
    package_name: str
    extra_args: Iterable[str] = ()
    required: bool = True


REQUIRED_DEPENDENCIES: List[Dependency] = [
    Dependency("numpy", "numpy"),
    Dependency("PIL", "Pillow"),
    Dependency("customtkinter", "customtkinter"),
    Dependency("tkinterdnd2", "tkinterdnd2", required=False),
    Dependency("cv2", "opencv-python"),
    Dependency("piexif", "piexif"),
    Dependency("skimage", "scikit-image"),
    # Optional quality-of-life extras. Installing them automatically keeps the
    # GUI consistent for first-time users but the application can run without
    # them if installation fails.
    Dependency("CTkToolTip", "CTkToolTip", required=False),
]


class DependencyInstallationError(RuntimeError):
    """Raised when automatic installation fails for any dependency."""

    def __init__(self, messages: Iterable[str]) -> None:
        self.messages: List[str] = list(messages)
        joined = "\n\n".join(self.messages)
        super().__init__(
            "One or more Python packages could not be installed automatically.\n"
            "Resolve the following issues manually and re-run the program:\n"
            f"{joined}"
        )


def _is_module_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def _install_dependency(dep: Dependency) -> None:
    cmd = [sys.executable, "-m", "pip", "install", dep.package_name, *dep.extra_args]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        return

    details = completed.stderr.strip() or completed.stdout.strip()
    message = (
        f"• Failed to install '{dep.package_name}'.\n"
        f"  Command: {' '.join(cmd)}\n"
        f"  Error: {details or 'Unknown error'}"
    )
    raise DependencyInstallationError([message])


def ensure_dependencies() -> None:
    """Install any missing third-party packages before the GUI imports run."""

    missing: List[Dependency] = [
        dep for dep in REQUIRED_DEPENDENCIES if not _is_module_available(dep.import_name)
    ]
    if not missing:
        return

    failures: List[str] = []
    optional_failures: List[str] = []
    for dep in missing:
        try:
            _install_dependency(dep)
        except DependencyInstallationError as exc:  # pragma: no cover - interactive usage
            messages = exc.messages if hasattr(exc, "messages") else [str(exc)]
            if dep.required:
                failures.extend(messages)
            else:
                optional_failures.extend(messages)

    if failures:
        raise DependencyInstallationError(failures)

    if optional_failures:
        optional_details = "\n\n".join(optional_failures)
        print(
            "Some optional Python packages could not be installed automatically.\n"
            "Optional features may be unavailable until they are installed manually.\n"
            f"{optional_details}",
            file=sys.stderr,
        )
