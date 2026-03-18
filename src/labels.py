"""
labels.py

Label registry loader and resolver for multi-dataset support.
Reads labels.json and provides class name resolution, binary mapping,
and dataset mode detection.

Author: Sanele Hlabisa
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


LABELS_PATH = Path(__file__).parent.parent / "labels.json"


class LabelRegistry:
    """
    Loads labels.json and provides lookup utilities.

    labels.json structure:
        multiclass: { "class name": [] }   ← canonical names, lowercase with spaces
        binary:     { "normal": [...], "abnormal": [...] }
    """

    def __init__(self, labels_path: Path = LABELS_PATH) -> None:
        with open(labels_path) as f:
            data = json.load(f)

        self.multiclass: dict[str, list[str]] = data["multiclass"]
        self.binary: dict[str, list[str]]     = data["binary"]

        # Sorted canonical class list — must match training order
        self.class_names: list[str] = sorted(self.multiclass.keys())
        self.class_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.class_names)
        }
        self.num_classes: int = len(self.class_names)

        # Reverse lookup: any alias/name → canonical name
        self._alias_to_canonical: dict[str, str] = {}
        for canonical in self.class_names:
            self._alias_to_canonical[canonical] = canonical

        # Build binary reverse lookup: canonical name → "normal" | "abnormal"
        self._canonical_to_binary: dict[str, str] = {}
        for verdict, names in self.binary.items():
            for name in names:
                self._canonical_to_binary[name.lower().strip()] = verdict

    def resolve(self, folder_name: str) -> Optional[str]:
        """
        Resolve a dataset folder name to a canonical class name.
        Returns None if not found — caller should warn.
        """
        key = folder_name.lower().strip()
        return self._alias_to_canonical.get(key)

    def to_idx(self, folder_name: str) -> Optional[int]:
        """Resolve folder name to integer class index."""
        canonical = self.resolve(folder_name)
        if canonical is None:
            return None
        return self.class_to_idx[canonical]

    def binary_verdict(self, canonical_name: str) -> Optional[str]:
        """Return 'normal' or 'abnormal' for a canonical class name."""
        return self._canonical_to_binary.get(canonical_name.lower().strip())

    def is_binary_dataset(self, folder_names: list[str]) -> bool:
        """
        Detect if a dataset should be evaluated in binary mode.
        True when the dataset has exactly 2 folders that both resolve
        to binary verdicts (e.g. 'violent' and 'non-violent').
        """
        if len(folder_names) != 2:
            return False
        verdicts = set()
        for name in folder_names:
            canonical = self.resolve(name)
            if canonical:
                verdict = self.binary_verdict(canonical)
                if verdict:
                    verdicts.add(verdict)
        return verdicts == {"normal", "abnormal"}

    def validate_folders(self, folder_names: list[str]) -> None:
        """
        Warn for any folder name that cannot be resolved.
        Raises ValueError if zero folders resolve successfully.
        """
        unresolved = []
        resolved = []
        for name in folder_names:
            if self.resolve(name) is None:
                unresolved.append(name)
            else:
                resolved.append(name)

        if unresolved:
            print(f"⚠️  Unresolved class folders (not in labels.json): {unresolved}")
            print(f"    Add them to labels.json multiclass and binary sections.")

        if not resolved:
            raise ValueError(
                "No dataset folders could be resolved against labels.json. "
                "Check that your dataset folder names match canonical names or aliases."
            )
