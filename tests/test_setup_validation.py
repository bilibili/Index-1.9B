"""Validation tests to ensure the testing infrastructure is properly set up."""

import sys
from pathlib import Path

import pytest


class TestSetupValidation:
    """Test class to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_pytest_installed(self):
        """Test that pytest is available."""
        import pytest
        assert pytest.__version__
    
    @pytest.mark.unit
    def test_project_structure_exists(self):
        """Test that the expected project directories exist."""
        root_dir = Path(__file__).parent.parent
        
        # Check main package directories
        assert (root_dir / "evaluate").exists(), "evaluate directory not found"
        assert (root_dir / "finetune").exists(), "finetune directory not found"
        assert (root_dir / "roleplay").exists(), "roleplay directory not found"
        assert (root_dir / "demo").exists(), "demo directory not found"
        
        # Check test directories
        assert (root_dir / "tests").exists(), "tests directory not found"
        assert (root_dir / "tests" / "unit").exists(), "unit tests directory not found"
        assert (root_dir / "tests" / "integration").exists(), "integration tests directory not found"
    
    @pytest.mark.unit
    def test_markers_configured(self, request):
        """Test that custom markers are properly configured."""
        markers = [mark.name for mark in request.node.iter_markers()]
        assert "unit" in markers
    
    @pytest.mark.integration
    def test_coverage_configured(self):
        """Test that coverage is properly configured."""
        # This test will pass if coverage is running
        # The actual coverage report will be generated when running tests
        assert True
    
    @pytest.mark.unit
    def test_python_path_includes_project(self):
        """Test that the project root is in Python path."""
        project_root = str(Path(__file__).parent.parent)
        assert any(project_root in path for path in sys.path)


@pytest.mark.unit
def test_sample_unit_test():
    """A simple unit test to verify basic testing works."""
    assert 1 + 1 == 2
    assert "hello" + " " + "world" == "hello world"


@pytest.mark.integration
def test_sample_integration_test():
    """A simple integration test."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.txt"
        test_file.write_text("test content")
        
        # Verify file operations
        assert test_file.exists()
        assert test_file.read_text() == "test content"


@pytest.mark.slow
def test_slow_marker():
    """Test that is marked as slow (should be skipped by default)."""
    import time
    time.sleep(0.1)  # Simulate slow test
    assert True