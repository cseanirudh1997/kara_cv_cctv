#!/usr/bin/env python3
"""
Test script for Ollama integration
"""

import requests
import json

def test_ollama_connection():
    """Test if Ollama is available and has Qwen model"""
    try:
        print("🔍 Testing Ollama connection...")
        
        # Test version endpoint
        r = requests.get("http://localhost:11434/api/version", timeout=3)
        if r.status_code == 200:
            print("✅ Ollama server is running")
            version_info = r.json()
            print(f"   Version: {version_info.get('version', 'unknown')}")
        else:
            print("❌ Ollama server not responding")
            return False
        
        # Test models endpoint
        r2 = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r2.status_code == 200:
            models = r2.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            print(f"📋 Available models: {len(model_names)}")
            for name in model_names:
                print(f"   - {name}")
            
            # Check for Qwen models
            qwen_models = [name for name in model_names if 'qwen' in name.lower()]
            if qwen_models:
                print(f"✅ Qwen models found: {qwen_models}")
                return True
            else:
                print("⚠️ No Qwen models found")
                print("💡 Run: ollama pull qwen2.5:7b")
                return False
        else:
            print("❌ Cannot list models")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Make sure Ollama is installed and running:")
        print("   1. Install from: https://ollama.ai")
        print("   2. Run: ollama serve")
        print("   3. Run: ollama pull qwen2.5:7b")
        return False

def test_simple_query():
    """Test a simple query to Qwen"""
    try:
        print("\n🧠 Testing Qwen 2.5 query...")
        
        payload = {
            "model": "qwen2.5:7b",
            "prompt": "Hello! Can you help with restaurant management questions?",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 2048
            }
        }
        
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        if r.status_code == 200:
            response = r.json()
            answer = response.get('response', '').strip()
            print(f"✅ Query successful!")
            print(f"📝 Response: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            return True
        else:
            print(f"❌ Query failed: HTTP {r.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Ollama Integration Test")
    print("=" * 50)
    
    connection_ok = test_ollama_connection()
    
    if connection_ok:
        query_ok = test_simple_query()
        if query_ok:
            print(f"\n🎉 Ollama integration is working perfectly!")
            print("✅ Ready for restaurant AI queries")
        else:
            print(f"\n⚠️ Connection works but queries are failing")
    else:
        print(f"\n❌ Ollama setup incomplete")
        
    print("\n" + "=" * 50)