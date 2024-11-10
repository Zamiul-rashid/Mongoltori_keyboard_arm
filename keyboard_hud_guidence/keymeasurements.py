import numpy as np

def calculate_apparent_measurements(distance_mm=1000, fov_degrees=96, image_width_px=1280, real_measurements=None):
    """Calculate apparent measurements at given distance"""
    
    # Calculate focal length based on FOV
    fov_rad = np.radians(fov_degrees)
    focal_length_px = (image_width_px/2) / np.tan(fov_rad/2)
    
    # Function to calculate apparent size
    def get_apparent_size(real_size):
        return (real_size * focal_length_px) / distance_mm
    
    # Default measurements at 0 meters if none provided
    if real_measurements is None:
        real_measurements = {
            'keyboard_width': 320,
            'keyboard_height': 140,
            'standard_key_size': 19,
            'key_spacing': 3.8,
            'spacebar_width': 120,
            'enter_width': 45,
            'shift_width': 43,
            'backspace_width': 38,
            'tab_width': 25,
            'caps_width': 28,
            'ctrl_width': 28,
            'alt_width': 28,
            'win_width': 28,
            'edge_margin': 7
        }
    
    # Calculate apparent measurements
    apparent_measurements = {
        key: get_apparent_size(value)
        for key, value in real_measurements.items()
    }
    
    # Calculate scaling factor (ratio of apparent to real size)
    scaling_factor = apparent_measurements['keyboard_width'] / real_measurements['keyboard_width']
    
    return apparent_measurements, scaling_factor

# Calculate and display measurements at different distances
distances = [1000]  # in mm

for distance in distances:
    print(f"\nMeasurements at {distance/1000:.1f} meter distance:")
    print("-" * 50)
    apparent_sizes, scaling = calculate_apparent_measurements(distance_mm=distance)
    
    print(f"Scaling factor: {scaling:.4f}")
    print("\nKey measurements (in pixels):")
    for key, value in apparent_sizes.items():
        original_mm = value/scaling
        print(f"{key:20}: {value:.1f}px (original: {original_mm:.1f}mm)")