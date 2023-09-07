from pathlib import Path


def get_data_home(data_home: Path | str | None = None) -> Path:
    """Returns the path of the cu_cat data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'cu_cat_data' in the
    user home folder.

    Alternatively, it can be set programmatically by giving an explicit folder
    path. The '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : pathlib.Path or string, optional
        The path to the cu_cat data directory. If `None`, the default path
        is `~/cu_cat_data`.

    Returns
    -------
    data_home : pathlib.Path
        The validated path to the cu_cat data directory.
    """
    if data_home is None:
        data_home = Path("~").expanduser() / "cu_cat_data"
    else:
        data_home = Path(data_home)
    data_home = data_home.resolve()
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def get_data_dir(name: str | None = None, data_home: Path | str | None = None) -> Path:
    """
    Returns the directory in which cu_cat looks for data.

    This is typically useful for the end-user to check
    where the data is downloaded and stored.

    Parameters
    ----------
    name : str, optional
        Subdirectory name. If omitted, the root data directory is returned.
    data_home : pathlib.Path or str, optional
        The path to cu_cat data directory. If `None`, the default path
        is `~/cu_cat_data`.
    """
    data_dir = get_data_home(data_home)
    if name is not None:
        data_dir = data_dir / name
    return data_dir
