"""Regression tests for immutable stroke-3 publication."""

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dataloaders.stroke_sequence_dataset import StrokeSequenceDataset
from prep_data import prepare_sketchformer


class Stroke3PublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.target = root / "target"
        self.source.mkdir()
        self._write_sources(1.0)

    def _write_sources(self, offset: float) -> None:
        for index, name in enumerate(("a", "b", "c")):
            stroke5 = np.asarray(
                [
                    [offset + index, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            np.savez(self.source / f"{name}.npz", stroke5=stroke5)

    def _prepare(self, chunks: int = 2) -> None:
        args = argparse.Namespace(
            source_dir=self.source,
            target_dir=self.target,
            n_chunks=chunks,
            seed=3,
            train_frac=1.0,
            valid_frac=0.0,
            test_frac=0.0,
        )
        with mock.patch.object(
            prepare_sketchformer, "parse_args", return_value=args
        ):
            prepare_sketchformer.main()

    def _active_chunks(self) -> list[Path]:
        with np.load(self.target / "meta.npz", allow_pickle=True) as meta:
            self.assertTrue(bool(meta["is_published"]))
            return [self.target / str(name) for name in meta["train_chunks"]]

    def test_rebuild_keeps_previous_version_usable(self) -> None:
        self._prepare()
        previous_chunks = self._active_chunks()
        previous_bytes = [path.read_bytes() for path in previous_chunks]
        previous_version = previous_chunks[0].parent
        self.assertEqual(
            len(StrokeSequenceDataset(previous_version, split="train")), 3
        )

        self._write_sources(10.0)
        self._prepare(chunks=1)

        self.assertEqual(
            [path.read_bytes() for path in previous_chunks], previous_bytes
        )
        self.assertNotEqual(self._active_chunks(), previous_chunks)
        self.assertEqual(
            len(StrokeSequenceDataset(previous_version, split="train")), 3
        )
        self.assertEqual(
            len(StrokeSequenceDataset(self.target, split="train")), 3
        )

    def test_failed_metadata_swap_leaves_active_dataset_intact(self) -> None:
        self._prepare()
        previous_meta = (self.target / "meta.npz").read_bytes()
        previous_chunks = self._active_chunks()
        previous_bytes = [path.read_bytes() for path in previous_chunks]
        self._write_sources(20.0)

        real_replace = os.replace

        def interrupt_replace(
            src: str | os.PathLike[str], dest: str | os.PathLike[str]
        ) -> None:
            if Path(dest) == self.target / "meta.npz":
                raise OSError("interrupted before publication")
            real_replace(src, dest)

        with mock.patch.object(
            prepare_sketchformer.os, "replace", side_effect=interrupt_replace
        ):
            with self.assertRaisesRegex(OSError, "interrupted before publication"):
                self._prepare()

        self.assertEqual(
            (self.target / "meta.npz").read_bytes(), previous_meta
        )
        self.assertEqual(
            [path.read_bytes() for path in previous_chunks], previous_bytes
        )
        self.assertEqual(
            len(StrokeSequenceDataset(self.target, split="train")), 3
        )

    def test_legacy_metadata_is_retained_for_reopening(self) -> None:
        self.target.mkdir()
        sketches = [np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)]
        prepare_sketchformer.save_chunk(
            self.target / "train_000.npz", sketches, [0]
        )
        np.savez(
            self.target / "meta.npz",
            train_chunks=np.asarray(["train_000.npz"], dtype=object),
            train_chunk_counts=np.asarray([1], dtype=np.int32),
            n_samples_train=1,
            is_published=True,
        )

        self._prepare()

        archives = list((self.target / "versions").glob("legacy_*_meta.npz"))
        self.assertEqual(len(archives), 1)
        old = StrokeSequenceDataset(
            self.target,
            split="train",
            metadata_file=str(archives[0].relative_to(self.target)),
        )
        self.assertEqual(len(old), 1)
