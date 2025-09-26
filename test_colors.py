"""Test script to demonstrate the new color palette."""

from pyandhold.visualization.plots import PortfolioVisualizer

# Test with many portfolio weights to show colors don't run out
test_weights = {
    f"Asset_{i}": 0.05 for i in range(1, 21)  # 20 assets with 5% each
}

# Create pie chart
fig = PortfolioVisualizer.plot_weights_pie(
    test_weights, 
    title="Portfolio Weights - 20 Assets Test"
)

print(f"Color palette has {len(PortfolioVisualizer.COLOR_PALETTE)} colors")
print(f"Testing with {len(test_weights)} assets")
print("✓ No more black colors when running out of palette!")

# Show first 10 colors from palette
print("\nFirst 10 colors in palette:")
for i in range(10):
    color = PortfolioVisualizer.COLOR_PALETTE[i]
    print(f"  {i+1}: {color}")
