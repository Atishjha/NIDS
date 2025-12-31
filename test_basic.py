# save as: test_basic.py
import os
import sys
import json
import joblib
import numpy as np

print("="*60)
print("NIDS BASIC TEST SUITE")
print("="*60)

def test_model_exists():
    """Test if trained model exists"""
    print("\n1. Testing Model Files...")
    
    model_files = [
        ('models/basic_nids_model.pkl', 'Main Model'),
        ('models/feature_info.json', 'Feature Info'),
    ]
    
    all_exist = True
    for file, description in model_files:
        if os.path.exists(file):
            print(f"   ✓ {description}: {file}")
            # Check file size
            size = os.path.getsize(file) / 1024  # KB
            print(f"     Size: {size:.1f} KB")
        else:
            print(f"   ✗ {description}: {file} (MISSING)")
            all_exist = False
    
    return all_exist

def test_model_loading():
    """Test if model loads correctly"""
    print("\n2. Testing Model Loading...")
    
    try:
        model = joblib.load('models/basic_nids_model.pkl')
        print(f"   ✓ Model loaded successfully")
        print(f"     Type: {type(model).__name__}")
        print(f"     Features expected: {model.n_features_in_}")
        
        # Load feature info
        with open('models/feature_info.json', 'r') as f:
            feature_info = json.load(f)
            print(f"     Features in config: {len(feature_info['feature_names'])}")
        
        return True
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        return False

def test_predictions():
    """Test making predictions"""
    print("\n3. Testing Predictions...")
    
    try:
        model = joblib.load('models/basic_nids_model.pkl')
        
        # Create test samples
        n_features = model.n_features_in_
        
        # Normal traffic sample
        normal_sample = np.random.rand(1, n_features) * 10
        # Attack traffic sample (higher values)
        attack_sample = np.random.rand(1, n_features) * 1000
        
        # Predict
        normal_pred = model.predict(normal_sample)[0]
        normal_proba = model.predict_proba(normal_sample)[0]
        
        attack_pred = model.predict(attack_sample)[0]
        attack_proba = model.predict_proba(attack_sample)[0]
        
        print(f"   ✓ Predictions working")
        print(f"     Normal sample: {'Attack' if normal_pred else 'Normal'} "
              f"(Confidence: {max(normal_proba)*100:.1f}%)")
        print(f"     Attack sample: {'Attack' if attack_pred else 'Normal'} "
              f"(Confidence: {max(attack_proba)*100:.1f}%)")
        
        return True
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        return False

def test_dashboard():
    """Test dashboard functionality"""
    print("\n4. Testing Dashboard...")
    
    try:
        import flask
        print(f"   ✓ Flask installed: {flask.__version__}")
        
        # Check dashboard file
        if os.path.exists('dashboard.py'):
            print(f"   ✓ Dashboard script exists")
            
            # Read first few lines
            with open('dashboard.py', 'r') as f:
                lines = f.readlines()[:5]
                if any('Flask' in line or 'app' in line for line in lines):
                    print(f"   ✓ Dashboard appears to be Flask app")
                else:
                    print(f"   ⚠ Dashboard may not be properly configured")
            
            return True
        else:
            print(f"   ✗ Dashboard script not found")
            return False
            
    except ImportError:
        print(f"   ✗ Flask not installed")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    tests = [
        test_model_exists,
        test_model_loading,
        test_predictions,
        test_dashboard
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ✗ Test error: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{total} tests")
    print(f"Score: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED! Your NIDS is ready.")
        print("\nNext: Run the system:")
        print("1. Start dashboard: python dashboard.py")
        print("2. Open browser: http://localhost:5000")
        print("3. Start monitoring: python nids_complete.py")
    else:
        print("\n⚠️  Some tests failed. Check issues above.")
    
    return all(results)

if __name__ == "__main__":
    run_all_tests()