#!/usr/bin/env python3
"""
Comprehensive test for min/max position constraints.
Tests all feasible and infeasible scenarios to validate constraint handling.
"""

import numpy as np
import pandas as pd
import importlib
import sys

# Force reload of modules
modules_to_reload = [
    'pyandhold.optimization.optimizers',
    'pyandhold.optimization.constraints'
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"Reloading {module_name}...")
        importlib.reload(sys.modules[module_name])
    else:
        print(f"{module_name} not loaded yet")

from pyandhold.optimization.optimizers import PortfolioOptimizer
from pyandhold.optimization.constraints import ConstraintBuilder


def test_max_position_constraints():
    """Test maximum position constraints with feasible and infeasible scenarios."""
    
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE MAX POSITION CONSTRAINT TESTS")
    print("="*80)
    
    # Create test data
    test_data = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
    constraint_builder = ConstraintBuilder()
    
    test_results = []
    
    # Test 1: FEASIBLE - Max 30% with 4 assets (4 × 30% = 120% > 100%)
    print("\n1. Testing FEASIBLE Max Position: 30% limit with 4 assets")
    print("   Expected: SUCCESS (4 × 30% = 120% > 100% needed)")
    try:
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        max_constraint = constraint_builder.max_position_constraint(0.30)
        
        # Check constraint detection
        has_min_pos = optimizer._has_minimum_position_constraints({'max_position': max_constraint})
        print(f"   Detected as min_position constraint: {has_min_pos} (should be False)")
        
        weights = optimizer.optimize_sharpe(constraints={'max_position': max_constraint}, verbose=True)
        
        max_weight = max(weights.values())
        total_weight = sum(weights.values())
        constraint_satisfied = max_weight <= 0.30 + 1e-6
        
        result = "✅ SUCCESS" if constraint_satisfied else "❌ VIOLATION"
        print(f"   {result}: Max weight = {max_weight:.4f} (limit: 0.30)")
        print(f"   Total weight: {total_weight:.4f}")
        print(f"   Weights: {[f'{w:.3f}' for w in weights.values()]}")
        
        test_results.append({
            'test': 'Max 30% (feasible)',
            'success': constraint_satisfied,
            'max_weight': max_weight,
            'limit': 0.30
        })
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        test_results.append({
            'test': 'Max 30% (feasible)',
            'success': False,
            'error': str(e)
        })
    
    # Test 2: FEASIBLE - Max 25% with 4 assets (4 × 25% = 100% = 100% needed)
    print("\n2. Testing FEASIBLE Max Position: 25% limit with 4 assets")
    print("   Expected: SUCCESS (4 × 25% = 100% = 100% needed)")
    try:
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        max_constraint = constraint_builder.max_position_constraint(0.25)
        
        weights = optimizer.optimize_sharpe(constraints={'max_position': max_constraint}, verbose=True)
        
        max_weight = max(weights.values())
        total_weight = sum(weights.values())
        constraint_satisfied = max_weight <= 0.25 + 1e-6
        
        result = "✅ SUCCESS" if constraint_satisfied else "❌ VIOLATION"
        print(f"   {result}: Max weight = {max_weight:.4f} (limit: 0.25)")
        print(f"   Total weight: {total_weight:.4f}")
        print(f"   Weights: {[f'{w:.3f}' for w in weights.values()]}")
        
        test_results.append({
            'test': 'Max 25% (barely feasible)',
            'success': constraint_satisfied,
            'max_weight': max_weight,
            'limit': 0.25
        })
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        test_results.append({
            'test': 'Max 25% (barely feasible)',
            'success': False,
            'error': str(e)
        })
    
    # Test 3: INFEASIBLE - Max 20% with 4 assets (4 × 20% = 80% < 100% needed)
    print("\n3. Testing INFEASIBLE Max Position: 20% limit with 4 assets")
    print("   Expected: INFEASIBLE ERROR (4 × 20% = 80% < 100% needed)")
    try:
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        max_constraint = constraint_builder.max_position_constraint(0.20)
        
        weights = optimizer.optimize_sharpe(constraints={'max_position': max_constraint}, verbose=True)
        
        # If we get here, something is wrong - should have failed
        max_weight = max(weights.values())
        total_weight = sum(weights.values())
        
        print(f"   🚨 UNEXPECTED SUCCESS: Max weight = {max_weight:.4f} (limit: 0.20)")
        print(f"   Total weight: {total_weight:.4f}")
        print(f"   This should have been rejected as infeasible!")
        
        test_results.append({
            'test': 'Max 20% (infeasible)',
            'success': False,  # Success is wrong here
            'unexpected_success': True,
            'max_weight': max_weight,
            'limit': 0.20
        })
        
    except Exception as e:
        print(f"   ✅ CORRECTLY REJECTED: {e}")
        test_results.append({
            'test': 'Max 20% (infeasible)',
            'success': True,  # Correctly rejected
            'correctly_rejected': True,
            'error': str(e)
        })
    
    # Test 4: EXTREME INFEASIBLE - Max 10% with 4 assets (4 × 10% = 40% < 100% needed)
    print("\n4. Testing EXTREME INFEASIBLE Max Position: 10% limit with 4 assets")
    print("   Expected: INFEASIBLE ERROR (4 × 10% = 40% < 100% needed)")
    try:
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        max_constraint = constraint_builder.max_position_constraint(0.10)
        
        weights = optimizer.optimize_sharpe(constraints={'max_position': max_constraint}, verbose=True)
        
        # If we get here, something is wrong
        max_weight = max(weights.values())
        total_weight = sum(weights.values())
        
        print(f"   🚨 UNEXPECTED SUCCESS: Max weight = {max_weight:.4f} (limit: 0.10)")
        print(f"   Total weight: {total_weight:.4f}")
        
        test_results.append({
            'test': 'Max 10% (extreme infeasible)',
            'success': False,
            'unexpected_success': True,
            'max_weight': max_weight,
            'limit': 0.10
        })
        
    except Exception as e:
        print(f"   ✅ CORRECTLY REJECTED: {e}")
        test_results.append({
            'test': 'Max 10% (extreme infeasible)',
            'success': True,
            'correctly_rejected': True,
            'error': str(e)
        })
    
    return test_results


def test_min_position_constraints():
    """Test minimum position constraints with feasible and infeasible scenarios."""
    
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE MIN POSITION CONSTRAINT TESTS")
    print("="*80)
    
    constraint_builder = ConstraintBuilder()
    test_results = []
    
    # Test 1: FEASIBLE - Min 5% with 4 assets (4 × 5% = 20% < 100%)
    print("\n1. Testing FEASIBLE Min Position: 5% minimum with 4 assets")
    print("   Expected: SUCCESS (can have 4×5% = 20%, leaving 80% for extra allocation)")
    try:
        test_data = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        min_constraint = constraint_builder.min_position_constraint(0.05)
        
        # Check constraint detection
        has_min_pos = optimizer._has_minimum_position_constraints({'min_position': min_constraint})
        print(f"   Detected as min_position constraint: {has_min_pos} (should be True)")
        
        weights = optimizer.optimize_sharpe(constraints={'min_position': min_constraint}, verbose=True)
        
        # Analyze results
        active_weights = [w for w in weights.values() if w > 1e-6]
        min_active_weight = min(active_weights) if active_weights else 0
        total_weight = sum(weights.values())
        constraint_satisfied = min_active_weight >= 0.05 - 1e-6
        
        result = "✅ SUCCESS" if constraint_satisfied else "❌ VIOLATION"
        print(f"   {result}: Min active weight = {min_active_weight:.4f} (minimum: 0.05)")
        print(f"   Total weight: {total_weight:.4f}")
        print(f"   Active positions: {len(active_weights)}")
        print(f"   All weights: {[f'{w:.3f}' for w in weights.values()]}")
        
        test_results.append({
            'test': 'Min 5% (feasible)',
            'success': constraint_satisfied,
            'min_active_weight': min_active_weight,
            'minimum': 0.05,
            'active_positions': len(active_weights)
        })
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        test_results.append({
            'test': 'Min 5% (feasible)',
            'success': False,
            'error': str(e)
        })
    
    # Test 2: BORDERLINE FEASIBLE - Min 10% with 4 assets (4 × 10% = 40%)
    print("\n2. Testing BORDERLINE Min Position: 10% minimum with 4 assets")
    print("   Expected: SUCCESS (can have 4×10% = 40%, leaving 60% for extra allocation)")
    try:
        test_data = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        min_constraint = constraint_builder.min_position_constraint(0.10)
        
        weights = optimizer.optimize_sharpe(constraints={'min_position': min_constraint}, verbose=True)
        
        active_weights = [w for w in weights.values() if w > 1e-6]
        min_active_weight = min(active_weights) if active_weights else 0
        constraint_satisfied = min_active_weight >= 0.10 - 1e-6
        
        result = "✅ SUCCESS" if constraint_satisfied else "❌ VIOLATION"
        print(f"   {result}: Min active weight = {min_active_weight:.4f} (minimum: 0.10)")
        print(f"   Active positions: {len(active_weights)}")
        print(f"   All weights: {[f'{w:.3f}' for w in weights.values()]}")
        
        test_results.append({
            'test': 'Min 10% (borderline feasible)',
            'success': constraint_satisfied,
            'min_active_weight': min_active_weight,
            'minimum': 0.10,
            'active_positions': len(active_weights)
        })
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        test_results.append({
            'test': 'Min 10% (borderline feasible)',
            'success': False,
            'error': str(e)
        })
    
    # Test 3: TIGHT - Min 20% with 4 assets (4 × 20% = 80%)
    print("\n3. Testing TIGHT Min Position: 20% minimum with 4 assets")
    print("   Expected: CHALLENGING (4×20% = 80%, only 20% left for extra allocation)")
    try:
        test_data = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        min_constraint = constraint_builder.min_position_constraint(0.20)
        
        weights = optimizer.optimize_sharpe(constraints={'min_position': min_constraint}, verbose=True)
        
        active_weights = [w for w in weights.values() if w > 1e-6]
        min_active_weight = min(active_weights) if active_weights else 0
        constraint_satisfied = min_active_weight >= 0.20 - 1e-6
        
        result = "✅ SUCCESS" if constraint_satisfied else "❌ VIOLATION"
        print(f"   {result}: Min active weight = {min_active_weight:.4f} (minimum: 0.20)")
        print(f"   Active positions: {len(active_weights)}")
        print(f"   All weights: {[f'{w:.3f}' for w in weights.values()]}")
        
        test_results.append({
            'test': 'Min 20% (tight)',
            'success': constraint_satisfied,
            'min_active_weight': min_active_weight,
            'minimum': 0.20,
            'active_positions': len(active_weights)
        })
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        test_results.append({
            'test': 'Min 20% (tight)',
            'success': False,
            'error': str(e)
        })
    
    # Test 4: INFEASIBLE - Min 30% with 4 assets (4 × 30% = 120% > 100%)
    print("\n4. Testing INFEASIBLE Min Position: 30% minimum with 4 assets")
    print("   Expected: INFEASIBLE ERROR (4×30% = 120% > 100% available)")
    try:
        test_data = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        min_constraint = constraint_builder.min_position_constraint(0.30)
        
        weights = optimizer.optimize_sharpe(constraints={'min_position': min_constraint}, verbose=True)
        
        # If we get here, something is wrong
        active_weights = [w for w in weights.values() if w > 1e-6]
        min_active_weight = min(active_weights) if active_weights else 0
        
        print(f"   🚨 UNEXPECTED SUCCESS: Min active weight = {min_active_weight:.4f}")
        print(f"   This should have been rejected as infeasible!")
        print(f"   All weights: {[f'{w:.3f}' for w in weights.values()]}")
        
        test_results.append({
            'test': 'Min 30% (infeasible)',
            'success': False,
            'unexpected_success': True,
            'min_active_weight': min_active_weight,
            'minimum': 0.30
        })
        
    except Exception as e:
        print(f"   ✅ CORRECTLY REJECTED: {e}")
        test_results.append({
            'test': 'Min 30% (infeasible)',
            'success': True,
            'correctly_rejected': True,
            'error': str(e)
        })
    
    return test_results


def test_combined_min_max_constraints():
    """Test combined min and max position constraints."""
    
    print("\n" + "="*80)
    print("🔍 COMBINED MIN/MAX POSITION CONSTRAINT TESTS")
    print("="*80)
    
    constraint_builder = ConstraintBuilder()
    test_results = []
    
    # Test 1: FEASIBLE - Min 5%, Max 40% with 4 assets
    print("\n1. Testing FEASIBLE Combined: Min 5%, Max 40% with 4 assets")
    print("   Expected: SUCCESS (can allocate 5-40% per position)")
    try:
        test_data = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        
        min_constraint = constraint_builder.min_position_constraint(0.05)
        max_constraint = constraint_builder.max_position_constraint(0.40)
        
        weights = optimizer.optimize_sharpe(
            constraints={'min_position': min_constraint, 'max_position': max_constraint}, 
            verbose=True
        )
        
        active_weights = [w for w in weights.values() if w > 1e-6]
        min_active = min(active_weights) if active_weights else 0
        max_weight = max(weights.values())
        
        min_satisfied = min_active >= 0.05 - 1e-6
        max_satisfied = max_weight <= 0.40 + 1e-6
        both_satisfied = min_satisfied and max_satisfied
        
        result = "✅ SUCCESS" if both_satisfied else "❌ VIOLATION"
        print(f"   {result}: Min active = {min_active:.4f} (≥0.05), Max = {max_weight:.4f} (≤0.40)")
        print(f"   All weights: {[f'{w:.3f}' for w in weights.values()]}")
        
        test_results.append({
            'test': 'Combined Min 5% Max 40%',
            'success': both_satisfied,
            'min_active': min_active,
            'max_weight': max_weight
        })
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        test_results.append({
            'test': 'Combined Min 5% Max 40%',
            'success': False,
            'error': str(e)
        })
    
    # Test 2: INFEASIBLE - Min 20%, Max 15% with 4 assets (contradictory)
    print("\n2. Testing INFEASIBLE Combined: Min 20%, Max 15% with 4 assets")
    print("   Expected: INFEASIBLE ERROR (min > max is contradictory)")
    try:
        test_data = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
        optimizer = PortfolioOptimizer(test_data, optimizer='mars')
        
        min_constraint = constraint_builder.min_position_constraint(0.20)
        max_constraint = constraint_builder.max_position_constraint(0.15)
        
        weights = optimizer.optimize_sharpe(
            constraints={'min_position': min_constraint, 'max_position': max_constraint}, 
            verbose=True
        )
        
        # If we get here, something is wrong
        print(f"   🚨 UNEXPECTED SUCCESS: This should have been rejected!")
        print(f"   Weights: {[f'{w:.3f}' for w in weights.values()]}")
        
        test_results.append({
            'test': 'Combined Min 20% Max 15% (contradictory)',
            'success': False,
            'unexpected_success': True
        })
        
    except Exception as e:
        print(f"   ✅ CORRECTLY REJECTED: {e}")
        test_results.append({
            'test': 'Combined Min 20% Max 15% (contradictory)',
            'success': True,
            'correctly_rejected': True,
            'error': str(e)
        })
    
    return test_results


def run_comprehensive_position_tests():
    """Run all comprehensive position constraint tests."""
    
    print("🚀 COMPREHENSIVE POSITION CONSTRAINT VALIDATION")
    print("Testing all feasible and infeasible scenarios for min/max constraints...")
    
    # Run all test suites
    max_results = test_max_position_constraints()
    min_results = test_min_position_constraints()
    combined_results = test_combined_min_max_constraints()
    
    # Combine all results
    all_results = max_results + min_results + combined_results
    
    # Summary
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    success_count = 0
    total_count = 0
    
    for result in all_results:
        total_count += 1
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {result['test']}")
        
        if result['success']:
            success_count += 1
        
        # Additional details for failures
        if not result['success']:
            if 'error' in result:
                print(f"     Error: {result['error']}")
            elif result.get('unexpected_success'):
                print(f"     Issue: Unexpected success for infeasible constraint")
    
    print(f"\n📊 Overall Results: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print(f"🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print(f"   Position constraints are working correctly!")
        return True
    elif success_count > 0:
        print(f"⚠️  PARTIAL SUCCESS ({success_count}/{total_count})")
        print(f"   Some position constraints working, but issues remain")
        return False
    else:
        print(f"💥 ALL TESTS FAILED")
        print(f"   Position constraint implementation needs major fixes")
        return False


if __name__ == "__main__":
    run_comprehensive_position_tests()
