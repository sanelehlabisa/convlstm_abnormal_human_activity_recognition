"""
labels.py

Label registry for multi-dataset support.
Drives class resolution, index assignment, and detection post-processing.
All class definitions live in labels.json — no hardcoding elsewhere.

Author: Sanele Hlabisa
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

LABELS_PATH = Path(__file__).parent.parent / "labels.json"


class LabelRegistry:
    """
    Loads labels.json and provides:
      - Stable class index assignment (always from multiclass keys)
      - Folder name resolution via aliases
      - Detection mode: maps canonical class → normal/abnormal
      - Dataset mode detection: multiclass vs detection
    """

    def __init__(self, labels_path: Path = LABELS_PATH) -> None:
        with open(labels_path) as f:
            data = json.load(f)

        # ---- Canonical classes (model always uses these) ----
        # Keys are the canonical names, values are alias lists
        self._multiclass: dict[str, list[str]] = data["multiclass"]

        # Sorted so indices are stable across runs
        self.class_names: list[str] = sorted(self._multiclass.keys())
        self.class_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.class_names)
        }
        self.num_classes: int = len(self.class_names)

        # ---- Alias → canonical (folder names resolve through here) ----
        # Built from multiclass values (alias lists)
        self._alias_to_canonical: dict[str, str] = {}
        for canonical, aliases in self._multiclass.items():
            self._alias_to_canonical[canonical.lower().strip()] = canonical
            for alias in aliases:
                self._alias_to_canonical[alias.lower().strip()] = canonical

        # ---- Detection mapping: canonical class → "normal" | "abnormal" ----
        # detection section lists canonical class names under each verdict
        self._detection: dict[str, list[str]] = data.get("detection", {})
        self._class_to_verdict: dict[str, str] = {}
        for verdict, class_list in self._detection.items():
            for cls in class_list:
                self._class_to_verdict[cls.lower().strip()] = verdict

    # ------------------------------------------------------------------
    # Folder name resolution
    # ------------------------------------------------------------------

    def resolve(self, folder_name: str) -> Optional[str]:
        """
        Resolve a dataset folder name to a canonical class name.
        Returns None if unresolvable — caller should warn.
        """
        return self._alias_to_canonical.get(folder_name.lower().strip())

    # ------------------------------------------------------------------
    # Detection post-processing
    # ------------------------------------------------------------------

    def detection_verdict(self, canonical_name: str) -> Optional[str]:
        """
        Return 'normal' or 'abnormal' for a canonical class name.
        Used to convert multiclass predictions to detection output.
        """
        return self._class_to_verdict.get(canonical_name.lower().strip())

    # ------------------------------------------------------------------
    # Dataset mode detection
    # ------------------------------------------------------------------

    def detect_mode(self, folder_names: list[str]) -> str:
        """
        Return 'detection' if the dataset has exactly 2 folders that
        both resolve to detection verdicts (normal / abnormal aliases).
        Otherwise return 'multiclass'.
        """
        if len(folder_names) != 2:
            return "multiclass"

        verdicts = set()
        for fname in folder_names:
            canonical = self.resolve(fname)
            if canonical is not None:
                verdict = self.detection_verdict(canonical)
                if verdict:
                    verdicts.add(verdict)

        if verdicts == {"normal", "abnormal"}:
            return "detection"
        return "multiclass"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_folders(self, folder_names: list[str]) -> None:
        """
        Warn for unresolvable folders.
        Raises ValueError if none resolve — likely a labels.json mismatch.
        """
        unresolved = [f for f in folder_names if self.resolve(f) is None]
        resolved   = [f for f in folder_names if self.resolve(f) is not None]

        if unresolved:
            print(f"⚠️  Unresolved folders (not in labels.json multiclass aliases): {unresolved}")
            print(f"    Add aliases to labels.json multiclass section.")

        if not resolved:
            raise ValueError(
                "No dataset folders resolved against labels.json. "
                "Check folder names match canonical names or aliases in multiclass."
            )
