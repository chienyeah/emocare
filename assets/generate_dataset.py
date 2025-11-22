import os
import time
import google.generativeai as genai

# Configuration
TARGET_TOTAL = 1500
BATCH_SIZE = 30  # Generate 50 samples per request to stay within token limits
OUTPUT_FILE = "comfort_examples.jsonl"
PROMPT_FILE = "prompt_template.txt"
MODEL_NAME = "gemini-3-pro-preview"  # Fast and capable model

def count_lines(filename):
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def main():
    # 1. Setup API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set it using: set GEMINI_API_KEY=your_key_here (Windows Command Prompt)")
        print("Or: $env:GEMINI_API_KEY='your_key_here' (PowerShell)")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    # 2. Read Prompt Template
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"Error: {PROMPT_FILE} not found.")
        return

    # 3. Generation Loop
    current_count = count_lines(OUTPUT_FILE)
    print(f"Starting generation. Current count: {current_count}/{TARGET_TOTAL}")

    while current_count < TARGET_TOTAL:
        needed = TARGET_TOTAL - current_count
        this_batch = min(BATCH_SIZE, needed)
        
        print(f"Generating batch of {this_batch} samples...")
        
        # The prompt template includes literal braces for JSON examples, so using
        # str.format would treat them as placeholders. Perform a direct replace
        # instead to avoid KeyErrors from unexpected fields.
        prompt = prompt_template.replace("{batch_size}", str(this_batch))
        
        try:
            response = model.generate_content(prompt)
            text_output = response.text.strip()
            
            # Simple validation to ensure we only append valid lines (heuristically)
            valid_lines = []
            for line in text_output.split('\n'):
                clean_line = line.strip()
                if clean_line.startswith('{') and clean_line.endswith('}'):
                    valid_lines.append(clean_line)
            
            if valid_lines:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    for line in valid_lines:
                        f.write(line + "\n")
                
                added = len(valid_lines)
                current_count += added
                print(f"  -> Added {added} samples. Total: {current_count}/{TARGET_TOTAL}")
            else:
                print("  -> Warning: No valid JSON lines found in response.")
                print("  -> Retrying...")

            # Avoid hitting rate limits too hard
            time.sleep(2)

        except Exception as e:
            print(f"  -> Error: {e}")
            print("  -> Waiting 10 seconds before retrying...")
            time.sleep(10)

    print(f"Done! {TARGET_TOTAL} samples collected in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
