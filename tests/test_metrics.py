"""
Unit tests for evaluation metrics.

Tests cover:
- RMSE calculation
- Perfect reconstruction
- Dimension handling
- Degradation ratio (optional)
- Import verification
"""
import pytest
import numpy as np
from evaluation.metrics import rmse, spread


def test_rmse_calculation():
    """RMSE should compute correctly for known inputs."""
    # Create simple test case
    analysis = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    truth = np.array([
        [1.0, 2.0, 3.0],
        [4.1, 5.1, 6.1],
        [7.2, 8.2, 9.2],
    ])
    
    result = rmse(analysis, truth)
    
    # Expected RMSE per dimension
    # Dim 0: sqrt(mean([0, 0.1^2, 0.2^2])) = sqrt(0.05/3) ≈ 0.1291
    # Dim 1: same
    # Dim 2: same
    expected = np.array([
        np.sqrt(np.mean([0.0**2, 0.1**2, 0.2**2])),
        np.sqrt(np.mean([0.0**2, 0.1**2, 0.2**2])),
        np.sqrt(np.mean([0.0**2, 0.1**2, 0.2**2])),
    ])
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_rmse_perfect_reconstruction():
    """RMSE should be zero when analysis equals truth."""
    truth = np.random.randn(100, 3)
    analysis = truth.copy()
    
    result = rmse(analysis, truth)
    
    np.testing.assert_allclose(result, np.zeros(3), atol=1e-10)


def test_rmse_dimensions():
    """RMSE should return correct shape for Lorenz-63."""
    analysis = np.random.randn(500, 3)
    truth = np.random.randn(500, 3)
    
    result = rmse(analysis, truth)
    
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    assert np.all(result >= 0), "RMSE should be non-negative"


def test_degradation_ratio():
    """Test degradation ratio calculation (RMSE_CS2 / RMSE_CS1)."""
    # Simulate two different RMSE values
    rmse_cs1 = np.array([0.1, 0.15, 0.2])
    rmse_cs2 = np.array([0.3, 0.45, 0.6])
    
    # Compute degradation ratio
    degradation = rmse_cs2 / rmse_cs1
    
    expected = np.array([3.0, 3.0, 3.0])
    np.testing.assert_allclose(degradation, expected, rtol=1e-5)
    
    # Verify all components show degradation
    assert np.all(degradation > 1.0), "CS2 should have higher RMSE than CS1"


def test_metrics_imports():
    """All metric functions should be importable."""
    try:
        from evaluation.metrics import rmse
        from evaluation.metrics import spread
        from evaluation.metrics import crps
        from evaluation.metrics import print_metrics_table
    except ImportError as e:
        pytest.fail(f"Failed to import metrics: {e}")
    
    # Verify they are callable
    assert callable(rmse), "rmse should be callable"
    assert callable(spread), "spread should be callable"
    assert callable(crps), "crps should be callable"
    assert callable(print_metrics_table), "print_metrics_table should be callable"


def test_spread_calculation():
    """Test spread metric for ensemble variance."""
    # Create ensemble variance (time x dimensions)
    ensemble_var = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    
    result = spread(ensemble_var)
    
    # Spread = sqrt(mean variance over time)
    expected = np.sqrt(np.mean(ensemble_var, axis=0))
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
