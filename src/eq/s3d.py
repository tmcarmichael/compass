"""Parse compressed asset archive files.

Archive format parser stubbed in public release. The class defines
the interface for loading packed environment geometry.

Archives store compressed files  -  zone geometry, textures, etc.
Format: 12-byte header -> zlib-compressed data blocks -> directory near EOF.
"""

from __future__ import annotations

from pathlib import Path


class S3DArchive:
    """Read-only access to files within a compressed asset archive."""

    MAGIC = 0x20534650  # "PFS " little-endian
    DIR_CRC = 0x61580AC9  # CRC identifying the filename directory entry

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, tuple[int, int]] = {}  # name -> (offset, size)
        raise NotImplementedError(
            "S3DArchive parser stubbed in public release. Provide an environment-specific implementation."
        )

    def list_files(self) -> list[str]:
        """Return sorted list of filenames in the archive."""
        raise NotImplementedError("S3DArchive parser stubbed in public release.")

    def extract(self, filename: str) -> bytes:
        """Extract and decompress a file by name."""
        raise NotImplementedError("S3DArchive parser stubbed in public release.")

    def __contains__(self, filename: str) -> bool:
        raise NotImplementedError("S3DArchive parser stubbed in public release.")

    def __repr__(self) -> str:
        return f"S3DArchive({self.path.name!r}, {len(self._entries)} files)"
