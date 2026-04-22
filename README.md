# Automatic Pressure Advance Analysis

A Python tool for analyzing pressure advance test images. The script processes image sequences, detects edges and width profiles, and generates an interactive HTML dashboard with metrics, plots, and image previews.

## Features

- Processes grayscale images (`.jpg`, `.jpeg`, `.png`)
- Detects strong edges and computes width, min/max, and position data
- Smooths width values for stable analysis
- Computes a quality score for each test image
- Generates an interactive dashboard in `dashboard.html`
- Automatically opens the dashboard in the default browser

## Project Structure

- `main.py` - Main script for analysis and dashboard generation
- `dashboard.html` - Generated report for browser viewing
- `img/Testimg/img2/` - Default input folder for image data

## Installation

1. Install Python 3
2. Install dependencies:

```bash
pip install pillow numpy matplotlib plotly
```

## Usage

1. Place your test images in the `img/Testimg/img2/` folder.
2. Name each file so the filename before the extension is the pressure advance value, e.g. `0.64.jpeg`.
3. Run the script:

```bash
python main.py
```

4. After execution, `dashboard.html` will be generated and opened automatically in your browser.

## Configuration

- The default input folder is defined in `main.py` as `folder_path = "./img/Testimg/img2/"`.
- The smoothing factor can be adjusted using `smooth_factor = 30`.

## Dependencies

- Pillow
- numpy
- matplotlib
- plotly

## Notes

- Image filenames must be parseable as numeric values to determine the pressure advance.
- The dashboard includes overview metrics, plots, and image previews.
- You can modify the output folder or HTML layout in `main.py` if needed.
