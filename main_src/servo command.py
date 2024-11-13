from pathlib import Path
import json

def servo_command():
    try:
        # Use relative path from current file
        json_path = Path(__file__).parent / 'keyboard_positions.json'
        with open(json_path, 'r') as f:
            keyboard_data = json.load(f)
            
        # Print the loaded data
        print("Frame Center:", keyboard_data['frame_center'])
        print("\nKey Positions:")
        for key, data in keyboard_data['key_positions'].items():
            print(f"\nKey {key}:")
            print(f"Absolute position: {data['absolute_position']}")
            print(f"Relative position: {data['relative_position']}")
            print(f"Dimensions: {data['dimensions']}")
            
    except FileNotFoundError:
        print(f"Error: Could not find keyboard positions file at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return

servo_command()
