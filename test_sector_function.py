#!/usr/bin/env python3
"""
Quick test to verify sector constraint function implementation.
"""

import numpy as np
from pyandhold.optimization.constraints import ConstraintBuilder

def test_sector_constraint_function():
    """Test that the sector constraint function works correctly."""
    
    print("🔍 TESTING SECTOR CONSTRAINT FUNCTION")
    print("=" * 50)
    
    # Simple test case: 4 assets, 2 sectors
    # Assets: A, B, C, D
    # Sectors: sector1 = [A, B], sector2 = [C, D]
    # Indices: A=0, B=1, C=2, D=3
    
    sector_mapping = {
        'sector1': [0, 1],  # First two assets
        'sector2': [2, 3]   # Last two assets
    }
    
    sector_limits = {
        'sector1': (0.3, 0.7),  # 30-70%
        'sector2': (0.2, 0.5)   # 20-50%
    }
    
    constraint_builder = ConstraintBuilder()
    constraints = constraint_builder.sector_constraint(sector_mapping, sector_limits)
    
    print(f"Generated {len(constraints)} constraints")
    
    # Test case 1: Valid allocation
    print("\nTest 1: Valid allocation")
    weights = np.array([0.4, 0.2, 0.3, 0.1])  # sector1=60%, sector2=40%
    print(f"Weights: {weights}")
    print(f"Sector1 (indices 0,1): {weights[0] + weights[1]:.1%}")
    print(f"Sector2 (indices 2,3): {weights[2] + weights[3]:.1%}")
    
    for i, constraint in enumerate(constraints):
        result = constraint['fun'](weights)
        status = "✅ satisfied" if result >= 0 else "❌ violated"
        print(f"  Constraint {i+1}: {result:.4f} ({status})")
    
    # Test case 2: Invalid allocation (sector1 too high)
    print("\nTest 2: Invalid allocation (sector1 too high)")
    weights = np.array([0.8, 0.1, 0.05, 0.05])  # sector1=90%, sector2=10%
    print(f"Weights: {weights}")
    print(f"Sector1 (indices 0,1): {weights[0] + weights[1]:.1%}")
    print(f"Sector2 (indices 2,3): {weights[2] + weights[3]:.1%}")
    
    violations = 0
    for i, constraint in enumerate(constraints):
        result = constraint['fun'](weights)
        status = "✅ satisfied" if result >= 0 else "❌ violated"
        if result < 0:
            violations += 1
        print(f"  Constraint {i+1}: {result:.4f} ({status})")
    
    print(f"\nTotal constraint violations: {violations}")
    
    if violations > 0:
        print("✅ Constraint function correctly detects violations!")
    else:
        print("❌ Problem: Constraint function should have detected violations!")
    
    return violations > 0

if __name__ == "__main__":
    success = test_sector_constraint_function()
    if success:
        print("\n🎉 SECTOR CONSTRAINT FUNCTION WORKING!")
    else:
        print("\n💥 SECTOR CONSTRAINT FUNCTION FAILED!")
