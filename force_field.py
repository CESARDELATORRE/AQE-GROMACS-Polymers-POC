# -*- coding: utf-8 -*-
"""Module :mod:`aiida_gromacs.data.force_field`."""
from __future__ import annotations

import pathlib

from aiida.common.lang import type_check
from aiida.orm import FolderData


class ForceFieldData(FolderData):
    """Data plugin to represent a custom force field for GROMACS."""

    def __init__(self, filepath: str | pathlib.Path, **kwargs):
        """Construct a new instance.

        :param filepath: Path to the directory containing the force field files. Note that the ``.ff`` suffix will
            automatically be added to the directory name as that is what GROMACS expects for a force field directory.
        :raises TypeError: If ``filepath`` has invalid type.
        :raises FileNotFoundError: If ``filepath`` is not an existing directory.
        """
        super().__init__(**kwargs)
        type_check(filepath, (str, pathlib.Path))

        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)

        filepath = filepath.absolute()

        if not filepath.is_dir():
            raise FileNotFoundError(f'filepath `{filepath}` is not a directory.')

        if not filepath.suffix:
            path = filepath.with_suffix('.ff')
        elif filepath.suffix != '.ff':
            path = filepath.with_suffix('.ff')
        else:
            path = filepath

        self.base.repository.put_object_from_tree(str(filepath), path.name)

    @property
    def dirname(self) -> str:
        """Return the directory name of the force field.

        :returns: The name of the top level directory. Guaranteed to end with the ``.ff`` suffix.
        """
        object_names = self.base.repository.list_object_names()
        assert len(object_names) == 1
        return object_names[0]

    @property
    def name(self) -> str:
        """Return the name of the force field.

        :returns: The name of the top level directory without the ``.ff`` suffix.
        """
        return str(pathlib.Path(self.dirname).with_suffix(''))
