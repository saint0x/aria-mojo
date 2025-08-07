"""
Simple Mojo build test for ARIA-LLaMA system.
"""

from src.mojo.main_training_system import SystemConfiguration

fn main():
    """Execute a simple build test."""
    print("🚀 ARIA-LLAMA BUILD TEST")
    print("=" * 40)
    
    # Test basic configuration
    var config = SystemConfiguration(
        base_examples=1000,
        augmentation_multiplier=2,
        random_seed=42
    )
    
    print("✅ SystemConfiguration created successfully")
    print("- Base examples:", config.base_examples)
    print("- Augmentation multiplier:", config.augmentation_multiplier)
    print("- Random seed:", config.random_seed)
    print("=" * 40)
    print("🎉 Build test completed successfully!")