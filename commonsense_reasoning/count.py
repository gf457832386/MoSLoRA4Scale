import os
import json

def process_json_files(directory):
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Initialize counts
            true_count = 0
            false_count = 0

            # Count "flag" values
            for item in data:
                if "flag" in item:
                    if item["flag"] == True:
                        true_count += 1
                    elif item["flag"] == False:
                        false_count += 1

            # Calculate total and accuracy
            total_count = true_count + false_count
            if total_count > 0:
                accuracy = (true_count / total_count) * 100
            else:
                accuracy = 0

            # Print statistics for each file
            print(f"File: {filename}")
            print(f"True Count: {true_count}")
            print(f"False Count: {false_count}")
            print(f"Total Count: {total_count}")
            print(f"Accuracy: {accuracy:.2f}%\n")

def main():
    # Replace 'your_directory_path' with the path to the directory containing the JSON files
    directory_path = "/home/hello/gh_Codes/MoSLoRA4Scale/commonsense_reasoning/results/moslora-r4-a32-3e4-GPU0-123111/round_3"
    process_json_files(directory_path)

# Ensure the script only runs when executed directly
if __name__ == "__main__":
    main()

