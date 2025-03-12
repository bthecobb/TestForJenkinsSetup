import json

# Path to training data
train_data_path = "train_data_fixed.jsonl"

# Read and validate JSONL
with open(train_data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            entry = json.loads(line.strip())  # Load each line as JSON
            assert "prompt" in entry, f"❌ Missing 'prompt' field in line {i}"
            assert "completion" in entry, f"❌ Missing 'completion' field in line {i}"
            assert isinstance(entry["prompt"], str), f"❌ 'prompt' should be string (line {i})"
            assert isinstance(entry["completion"], str), f"❌ 'completion' should be string (line {i})"
            print(f"✅ Line {i}: OK")
        except Exception as e:
            print(f"❌ Error in line {i}: {e}")

print("🎯 Training data validation complete!")
