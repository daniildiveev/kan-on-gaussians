"""Fingerprint the source files installed by this project."""

import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class SourceVersion:
    root: Path = Path(__file__).resolve().parents[1]

    def tags(self) -> dict[str, str]:
        packages = ("data", "experiments", "models", "tracking", "training")
        files = [self.root / "config.py"]
        for name in packages:
            files.extend((self.root / name).rglob("*.py"))

        digest = sha256()
        for path in sorted(files):
            relative = path.relative_to(self.root).as_posix()
            digest.update(relative.encode() + b"\0")
            digest.update(sha256(path.read_bytes()).digest())
        tags = {"mgs_kan.source_sha256": digest.hexdigest()}

        if self.root.name == "src" and (self.root.parent / ".git").exists():
            try:
                commit = self._git("rev-parse", "HEAD")
                changes = self._git("--no-optional-locks", "status", "--porcelain", "--",
                                    "config.py", *packages)
            except (OSError, subprocess.CalledProcessError):
                return tags
            tags["mlflow.source.git.commit"] = commit
            tags["mgs_kan.source_dirty"] = "true" if changes else "false"
        return tags

    def _git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=self.root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
