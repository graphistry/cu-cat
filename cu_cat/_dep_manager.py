import importlib


class DepManager:
    """Easily keep track of dependencies in a project.
    Parameters
    ----------
    package : str
        The name of the package to import.
    
    Attributes
    ----------
    pkgs : dict
        A dictionary of the imported packages.
        
    Examples
    --------
    >>> numpy = deps.numpy
    """
    def __init__(self):
        self.pkgs = {}

    def __getattr__(self, pkg: str):
        self._add_deps(pkg)
        try:
            return self.pkgs[pkg]
        except KeyError:
            return None

    def _add_deps(self, pkg: str):
        try:
            pkg_val = importlib.import_module(pkg)
            self.pkgs[pkg] = pkg_val
            setattr(self, pkg, pkg_val)
        except:
            pass

    def import_from(self, pkg: str, name: str):
        try:
            module = __import__(pkg, fromlist=[name])
            self.pkgs[name] = module
        except:
            pass


deps = DepManager()
