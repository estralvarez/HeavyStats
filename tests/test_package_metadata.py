import pathlib
import heavystats as hs


def test_package_version():
    assert hs.__version__ == "0.3.0"


def test_package_exports():
    assert hasattr(hs, "validate_data")
    assert hasattr(hs, "ValidationReport")
    assert hasattr(hs, "compare_groups")
    assert hasattr(hs, "ComparationReport")
    assert hasattr(hs, "load_data")
    assert hasattr(hs, "UnivariateTables")
    assert hasattr(hs, "UnivariatePlots")
    assert hasattr(hs, "BivariateTables")
    assert hasattr(hs, "BivariatePlots")
    assert hasattr(hs, "mann_whitney_test")
    assert hasattr(hs, "kruskal_wallis_test")
    assert hasattr(hs, "spearman_matrix")


def test_bundled_data_exists():
    pkg_dir = pathlib.Path(hs.__file__).parent
    data_file = pkg_dir / "data" / "data_example.csv"
    assert data_file.exists(), f"El archivo de datos de ejemplo no existe en {data_file}"
    assert data_file.stat().st_size > 0


def test_py_typed_exists():
    pkg_dir = pathlib.Path(hs.__file__).parent
    py_typed = pkg_dir / "py.typed"
    assert py_typed.exists(), f"py.typed no existe en {py_typed}"
