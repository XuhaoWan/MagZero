import magzero


def test_package_version_is_exposed():
    assert hasattr(magzero, "__version__")
    assert isinstance(magzero.__version__, str)
