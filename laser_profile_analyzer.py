import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_laser_intensity_data(resolution=100):
    """
    Simulates raw, uncleaned matrix data from a laser beam profile sensor,
    incorporating environmental background signal noise.
    """
    x = np.linspace(-3, 3, resolution)
    y = np.linspace(-3, 3, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Mathematical expression for a standard TEM00 Gaussian beam profile
    clean_profile = np.exp(-(X**2 + Y**2))
    
    # Introduce random environmental fluctuations/noise to represent raw data
    ambient_noise = np.random.normal(0, 0.12, (resolution, resolution))
    raw_experimental_matrix = clean_profile + ambient_noise
    
    return X, Y, raw_experimental_matrix

def analyze_and_clean_profile(raw_data):
    """
    Applies data filtering arrays to eliminate noise and isolate the core beam center.
    """
    # Isolate and nullify extreme edge background noise (thresholding)
    cleaned_matrix = np.where(raw_data < 0.05, 0, raw_data)
    
    # Apply a Gaussian low-pass spatial filter to smooth variations
    smoothed_matrix = gaussian_filter(cleaned_matrix, sigma=1.5)
    
    return smoothed_matrix

def main():
    print("[SYSTEM] Initializing Computational Laser Profile Run...")
    
    # 1. Fetch raw sensor matrix information
    X, Y, raw_data = generate_laser_intensity_data(resolution=150)
    
    # 2. Execute computational noise removal algorithms
    processed_beam = analyze_and_clean_profile(raw_data)
    
    # 3. Calculate primary metrics
    peak_intensity = np.max(processed_beam)
    print(f"[METRIC SUCCESS] Extracted Peak Intensity: {peak_intensity:.4f} W/cm²")
    
    # 4. Generate Scientific Data Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw Data Charting
    im1 = axes[0].pcolormesh(X, Y, raw_data, cmap='inferno', shading='auto')
    axes[0].set_title("1. Raw Laser Profile Matrix (With Noise)")
    axes[0].set_xlabel("X-Axis (mm)")
    axes[0].set_ylabel("Y-Axis (mm)")
    fig.colorbar(im1, ax=axes[0], label='Intensity Scale')
    
    # Processed Data Contour Mapping
    im2 = axes[1].contourf(X, Y, processed_beam, levels=15, cmap='viridis')
    axes[1].set_title("2. Cleaned Mathematical Contour Mapping")
    axes[1].set_xlabel("X-Axis (mm)")
    axes[1].set_ylabel("Y-Axis (mm)")
    fig.colorbar(im2, ax=axes[1], label='Intensity Scale')
    
    plt.tight_layout()
    print("[SYSTEM] Generating scientific visualization window...")
    plt.show()

if __name__ == "__main__":
    main()
  
