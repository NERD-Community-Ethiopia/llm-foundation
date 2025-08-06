#!/usr/bin/env python3
"""
Real Data Testing Runner
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🚀 Running Real Data Attention Tests\n")
    
    try:
        # Import the real data testing module
        from attention.examples.real_data_testing import run_real_data_tests
        from attention.examples.text_classification_example import test_text_classification
        
        print("=" * 80)
        print("🧠 REAL DATA ATTENTION TESTING")
        print("=" * 80)
        
        # Run real data tests
        run_real_data_tests()
        
        print("\n" + "=" * 80)
        print("📚 TEXT CLASSIFICATION WITH ATTENTION")
        print("=" * 80)
        
        # Run text classification test
        test_text_classification()
        
        print("\n" + "=" * 80)
        print("✅ ALL REAL DATA TESTS COMPLETED SUCCESSFULLY!")
        print("📊 Check the generated visualizations to see attention in action!")
        print("🎯 This demonstrates attention mechanisms with actual text data!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all attention modules are properly implemented")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 