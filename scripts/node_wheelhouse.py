"""Build the hash-locked, CPython 3.12 Linux node dependency payload."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pip._vendor.packaging.version import Version


LOCKS = ("node-runtime.lock", "node-tools.lock")
PLATFORM = {
    "os": "linux",
    "machine": "x86_64",
    "implementation": "cpython",
    "python": [3, 12],
    "minimum_glibc": [2, 17],
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class NodeImageCache:
    """Authenticated lookup hints; wheel authorization still comes from locks/OCI."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = (
            directory
            if directory is not None
            else Path.home() / ".cache/gpu-fault/node-dependency-images-v1"
        ).absolute()

    def _directory(self, *, create: bool) -> bool:
        for path in (*reversed(self.directory.parents), self.directory):
            try:
                info = path.lstat()
            except FileNotFoundError:
                if not create:
                    return False
                try:
                    path.mkdir(mode=0o700)
                except FileExistsError:
                    pass
                info = path.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid not in {0, os.geteuid()}
                or (
                    info.st_mode & 0o022
                    and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX)
                )
            ):
                raise ValueError("node image cache directory is not trusted")
        info = self.directory.stat()
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("node image cache must be owner-private")
        return True

    @staticmethod
    def _check_file(info: os.stat_result) -> None:
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
        ):
            raise ValueError("node image cache file is not owner-private and regular")

    @contextmanager
    def _locked(self, *, create: bool) -> Iterator[bool]:
        if not self._directory(create=create):
            yield False
            return
        descriptor = os.open(
            self.directory / ".lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        try:
            self._check_file(os.fstat(descriptor))
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield True
        finally:
            os.close(descriptor)

    def _read(self, path: Path, *, limit: int) -> bytes:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            self._check_file(os.fstat(descriptor))
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                content = stream.read(limit + 1)
        finally:
            os.close(descriptor)
        if len(content) > limit:
            raise ValueError("node image cache file exceeds its size limit")
        return content

    def _write(self, path: Path, content: bytes) -> None:
        descriptor, name = tempfile.mkstemp(prefix=".receipt-", dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _key(self, *, create: bool) -> bytes:
        path = self.directory / ".receipt-key"
        try:
            key = self._read(path, limit=32)
        except FileNotFoundError:
            if not create or any(self.directory.glob("*.json")):
                raise ValueError(
                    "node image cache authentication key is missing"
                ) from None
            key = os.urandom(32)
            self._write(path, key)
        if len(key) != 32:
            raise ValueError("node image cache authentication key is invalid")
        return key

    @staticmethod
    def _encoded(value: object) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def _load(self, identity: str) -> dict[str, Any] | None:
        try:
            content = self._read(self.directory / f"{identity}.json", limit=131072)
        except FileNotFoundError:
            return None
        key = self._key(create=False)
        try:
            envelope = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("node image cache receipt is invalid") from exc
        if not isinstance(envelope, dict) or set(envelope) != {
            "receipt",
            "hmac_sha256",
        }:
            raise ValueError("node image cache receipt is invalid")
        receipt = envelope["receipt"]
        signature = envelope["hmac_sha256"]
        if (
            not isinstance(signature, str)
            or not re.fullmatch(r"[0-9a-f]{64}", signature)
            or not hmac.compare_digest(
                hmac.digest(key, self._encoded(receipt), "sha256").hex(), signature
            )
            or not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("identity") != identity
            or not isinstance(receipt.get("image"), dict)
        ):
            raise ValueError("node image cache receipt authentication failed")
        return dict(receipt["image"])

    def load(self, identity: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError("node image cache identity is invalid")
        with self._locked(create=False) as available:
            if not available:
                return None
            if (
                os.path.lexists(self.directory / ".receipt-key")
                or next(self.directory.glob("*.json"), None) is not None
            ):
                # A miss must not hide a damaged authentication key until after
                # the expensive download/build/push has already completed.
                self._key(create=False)
            return self._load(identity)

    def store(self, identity: str, image: Mapping[str, Any]) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError("node image cache identity is invalid")
        with self._locked(create=True):
            self._load(identity)
            key = self._key(create=True)
            receipt = {"schema_version": 1, "identity": identity, "image": dict(image)}
            envelope = {
                "receipt": receipt,
                "hmac_sha256": hmac.digest(key, self._encoded(receipt), "sha256").hex(),
            }
            self._write(
                self.directory / f"{identity}.json", self._encoded(envelope) + b"\n"
            )


def validate_build_host() -> None:
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise ValueError("node wheelhouse requires a Linux Python 3.12 build host")


def _locked_wheel_hashes(
    locks: Mapping[str, bytes],
) -> dict[str, tuple[Version, set[str]]]:
    from pip._internal.req.req_file import break_args_options, preprocess
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version

    expected: dict[str, tuple[Version, set[str]]] = {}
    for content in locks.values():
        text = content.decode("utf-8")
        if "${" in text:
            raise ValueError("environment expansion is forbidden in node locks")
        seen: set[str] = set()
        for _line_number, line in preprocess(text):
            requirement, options = break_args_options(line)
            hashes = shlex.split(options)
            if not hashes or any(
                not re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", value)
                for value in hashes
            ):
                raise ValueError("node lock requires only pinned SHA-256 hashes")
            parsed = Requirement(requirement)
            pins = list(parsed.specifier)
            if (
                parsed.url
                or parsed.extras
                or len(pins) != 1
                or pins[0].operator != "=="
                or "*" in pins[0].version
            ):
                raise ValueError("node lock must contain exact package versions")
            if parsed.marker is not None and not parsed.marker.evaluate():
                continue
            name = canonicalize_name(parsed.name)
            version = Version(pins[0].version)
            allowed = {value.removeprefix("--hash=sha256:") for value in hashes}
            if name in seen:
                raise ValueError("duplicate applicable node lock requirement")
            seen.add(name)
            if name in expected:
                previous_version, previous_hashes = expected[name]
                if version != previous_version:
                    raise ValueError("node locks require conflicting package versions")
                allowed &= previous_hashes
                if not allowed:
                    raise ValueError("node locks require conflicting wheel hashes")
            expected[name] = version, allowed
    if not expected:
        raise ValueError("node locks have no applicable wheel requirements")
    return expected


def _inventory_bytes(inventory: object) -> bytes:
    return (
        json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _validate_wheelhouse_inventory(root: Path, inventory: object) -> str:
    """Bind every inventoried wheel to the current locks, not a cache's own hash."""
    from pip._internal.utils.compatibility_tags import get_supported
    from pip._vendor.packaging.utils import parse_wheel_filename

    locks = {name: (root / "requirements" / name).read_bytes() for name in LOCKS}
    lock_hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in locks.items()
    }
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"schema_version", "platform", "locks", "files"}
        or type(inventory["schema_version"]) is not int
        or inventory["schema_version"] != 1
        or inventory["platform"] != PLATFORM
        or inventory["locks"] != lock_hashes
        or not isinstance(inventory["files"], dict)
        or not set(LOCKS) <= set(inventory["files"])
    ):
        raise ValueError("node wheelhouse inventory does not match current locks")
    expected = _locked_wheel_hashes(locks)
    supported = set(
        get_supported(
            version="312",
            platforms=["manylinux2014_x86_64"],
            impl="cp",
            abis=["cp312"],
        )
    )
    observed: set[str] = set()
    for filename, entry in inventory["files"].items():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(entry, dict)
            or set(entry) != {"sha256", "size"}
            or not isinstance(entry["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or type(entry["size"]) is not int
            or entry["size"] < 0
        ):
            raise ValueError("node wheelhouse inventory file entry is invalid")
        if filename in locks:
            if entry["sha256"] != lock_hashes[filename] or entry["size"] != len(
                locks[filename]
            ):
                raise ValueError("node wheelhouse inventory lock file differs")
            continue
        name, version, _build, tags = parse_wheel_filename(filename)
        if (
            name not in expected
            or version != expected[name][0]
            or entry["sha256"] not in expected[name][1]
            or not tags & supported
            or entry["size"] == 0
            or name in observed
        ):
            raise ValueError(
                "node wheel is not bound to current hash-locked requirements"
            )
        observed.add(name)
    if observed != set(expected):
        raise ValueError("node wheelhouse does not contain every locked requirement")
    return hashlib.sha256(_inventory_bytes(inventory)).hexdigest()


def validate_wheelhouse_inventory(root: Path, inventory: object) -> str:
    """Keep pip's process-global logging/plugin setup outside the build driver."""
    try:
        payload = json.dumps(inventory, sort_keys=True)
    except (TypeError, ValueError):
        raise ValueError("node wheelhouse inventory is not serializable") from None
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), str(root.resolve())],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("node wheelhouse inventory validation timed out") from None
    try:
        report = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise ValueError(
            "node wheelhouse validator returned an invalid report"
        ) from None
    if not isinstance(report, dict):
        raise ValueError("node wheelhouse validator returned an invalid report")
    if result.returncode:
        raise ValueError(
            str(report.get("error") or "node wheelhouse inventory validation failed")
        )
    digest = report.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("node wheelhouse validator returned an invalid digest")
    return digest


def build_wheelhouse(
    root: Path,
    destination: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    expected_locks: Mapping[str, str] | None = None,
) -> str:
    validate_build_host()
    locks = {name: sha256(root / "requirements" / name) for name in LOCKS}
    if expected_locks is not None and locks != expected_locks:
        raise ValueError("node dependency locks changed before download")
    destination.mkdir(parents=True)
    destination.chmod(0o755)
    for name in LOCKS:
        lock = root / "requirements" / name
        copied_lock = destination / name
        shutil.copyfile(lock, copied_lock)
        if sha256(copied_lock) != locks[name]:
            raise ValueError("node dependency lock changed during copy")
    for name in LOCKS:
        copied_lock = destination / name
        command = [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "--isolated",
            "download",
            "--disable-pip-version-check",
            "--require-hashes",
            "--only-binary=:all:",
            "--platform",
            "manylinux2014_x86_64",
            "--python-version",
            "3.12",
            "--implementation",
            "cp",
            "--abi",
            "cp312",
            "--dest",
            str(destination),
            "--requirement",
            str(copied_lock),
        ]
        result = runner(command, check=False, text=True)
        if result.returncode:
            raise ValueError(f"hash-locked node dependency download failed: {name}")
    for name in LOCKS:
        if (
            sha256(root / "requirements" / name) != locks[name]
            or sha256(destination / name) != locks[name]
        ):
            raise ValueError("node dependency lock changed during download")
    wheels = list(destination.glob("*.whl"))
    if not wheels:
        raise ValueError("node dependency wheelhouse is empty")
    files = {
        path.name: {"sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(destination.iterdir())
        if path.is_file() and not path.is_symlink()
    }
    if len(files) != len(list(destination.iterdir())):
        raise ValueError("node dependency wheelhouse contains non-regular files")
    for path in destination.iterdir():
        path.chmod(0o644)
    manifest = {
        "schema_version": 1,
        "platform": PLATFORM,
        "locks": locks,
        "files": files,
    }
    inventory_sha256 = validate_wheelhouse_inventory(root, manifest)
    inventory = destination / "inventory.json"
    inventory.write_bytes(_inventory_bytes(manifest))
    inventory.chmod(0o644)
    if sha256(inventory) != inventory_sha256:
        raise ValueError("node wheelhouse inventory changed during publication")
    return inventory_sha256


def _main(root: Path) -> int:
    try:
        value = _validate_wheelhouse_inventory(root, json.load(sys.stdin))
    except OSError:
        report = {"error": "node wheelhouse lock input is unreadable"}
    except ValueError as exc:
        report = {
            "error": str(exc)
            if type(exc) is ValueError
            else "node wheelhouse contains invalid locked requirements or wheel filenames"
        }
    else:
        print(json.dumps({"sha256": value}))
        return 0
    print(json.dumps(report))
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("node wheelhouse validator requires one source root")
    raise SystemExit(_main(Path(sys.argv[1])))
