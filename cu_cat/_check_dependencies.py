<<<<<<< HEAD
from importlib.metadata import PackageNotFoundError, requires, version

from packaging.requirements import Requirement


def check_dependencies():
    package_name = "cu_cat"
    package_version = version(package_name)
    requirements = requires(package_name)

    for req in requirements:
        req = Requirement(req)

        if req.marker is not None:
            # skip extra requirements
            continue

        try:
            installed_dep = version(req.name)
            if not req.specifier.contains(installed_dep, prereleases=True):
                raise ImportError(
                    f"{package_name} {package_version} requires {req!s} but you have"
                    f" {req.name} {installed_dep} installed, which is incompatible."
                )

        except PackageNotFoundError:
            raise ImportError(
                f"{package_name} {package_version} requires {req!s}, "
                "which you don't have installed."
=======
import pkg_resources


def check_dependencies():
    package_name = "cu-cat"
    env = pkg_resources.Environment()
    package = env[package_name][0]
    requirements = package.requires()
    for req in requirements:
        try:
            installed_dep = next(
                iter(
                    (
                        installed_dep
                        for installed_dep in env[req.name]
                        if installed_dep.project_name == req.name
                    )
                )
            )
        except StopIteration:
            raise ImportError(
                f"{package_name} {package.version} requires {req}, "
                "which you don't have."
            )

        if installed_dep not in req:
            raise ImportError(
                f"{package_name} {package.version} requires {req} "
                f"but you have {installed_dep}, which is not compatible."
>>>>>>> cu-cat/DT5
            )
