from freshrank import __version__


def test_package_imports() -> None:
    """Verify that the minimal project and test configuration are wired correctly."""
    assert __version__ == "0.1.0"
