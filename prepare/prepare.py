import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_grayscale(image_path: Path) -> np.ndarray:
    """Load image and rotate if necessary for horizontal lines."""
    image = Image.open(image_path).convert('L')
    # Auto-detect orientation: if width > height, rotate to make lines horizontal
    if image.width > image.height:
        image = image.rotate(-90, expand=True)
    return np.asarray(image, dtype=np.uint8)


def find_horizontal_lines(image_array: np.ndarray) -> list[int]:
    """Find horizontal line centers using smoothed row means and clustering."""
    # Calculate mean brightness for each row (horizontal lines)
    row_mean = image_array.mean(axis=1)
    
    # Smooth the signal to reduce noise
    kernel_size = 21
    smooth = np.convolve(row_mean, np.ones(kernel_size)/kernel_size, mode='same')
    
    # Find regions above threshold (bright lines)
    threshold = np.mean(smooth) + 1.5 * np.std(smooth)
    active_rows = np.where(smooth > threshold)[0]
    
    # Cluster consecutive active rows into line regions
    clusters = []
    if len(active_rows) > 0:
        start = active_rows[0]
        prev = active_rows[0]
        
        for i in active_rows[1:]:
            if i - prev > 3:  # Gap ends cluster
                clusters.append((start, prev))
                start = i
            prev = i
        clusters.append((start, prev))  # Add last cluster
    
    # Return cluster centers
    centers = [(s + e) // 2 for s, e in clusters]
    return centers


def crop_line_region(image: Image.Image, line_center: int, line_height: int = 70) -> Image.Image:
    """Crop a region around a horizontal line center."""
    half_height = line_height // 2
    top = max(0, line_center - half_height)
    bottom = min(image.height, line_center + half_height)
    return image.crop((0, top, image.width, bottom))


def prepare_image(image_path: Path, output_dir: Path, start_value: float = 0.0, step: float = 0.002, line_height: int = 70, debug: bool = False) -> int:
    """Prepare ORCA pressure advance test image by extracting horizontal lines."""
    # Load and process image
    array = load_grayscale(image_path)
    line_centers = find_horizontal_lines(array)
    
    if len(line_centers) == 0:
        print(f"Warning: No lines found in {image_path.name}")
        return 0
    
    print(f"Found {len(line_centers)} horizontal lines in {image_path.name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_dir / image_path.name)

    # Load image for cropping (with same rotation)
    image = Image.open(image_path).convert('RGB')
    if image.width > image.height:
        image = image.rotate(-90, expand=True)
    
    # Sort centers from bottom to top (reverse order since PIL y=0 is top)
    line_centers_sorted = sorted(line_centers, reverse=True)
    
    # Extract and save each line
    line_count = 0
    for idx, center in enumerate(line_centers_sorted):
        pa_value = start_value + idx * step
        line_crop = crop_line_region(image, center, line_height)
        out_name = output_dir / f"{pa_value:.3f}.png"
        line_crop.save(out_name)
        line_count += 1

    # Create debug annotation if requested
    if debug:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for center in line_centers:
            draw.line([0, center, image.width, center], fill=(255, 0, 0), width=2)
        annotated.save(output_dir / f"{image_path.stem}_annotated.png")

    print(f"Prepared {image_path.name}: {line_count} line images saved")
    return line_count


def process_path(source: Path, output_root: Path, start_value: float = 0.0, step: float = 0.002, line_height: int = 70, debug: bool = False) -> None:
    if source.is_dir():
        images = sorted([p for p in source.iterdir() if is_image_file(p)])
        if not images:
            raise ValueError(f"Kein Bild im Verzeichnis gefunden: {source}")
        for image_path in images:
            subfolder = output_root / f"{image_path.stem}_lines"
            prepare_image(image_path, subfolder, start_value, step, line_height, debug=debug)
    elif is_image_file(source):
        subfolder = output_root / f"{source.stem}_lines"
        prepare_image(source, subfolder, start_value, step, line_height, debug=debug)
    else:
        raise ValueError(f"Pfad ist kein gültiges Bild oder Verzeichnis: {source}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare ORCA pressure advance test images by splitting each line into separate images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('source', help='Bilddatei oder Ordner mit Bildern')
    parser.add_argument('--output', help='Zielordner für vorbereitete Zeilenbilder', default=None)
    parser.add_argument('--start-value', type=float, default=0.0, 
                       help='Startwert für Pressure Advance (z.B. 0.0)')
    parser.add_argument('--step', type=float, default=0.002,
                       help='Schrittweite für Pressure Advance (z.B. 0.002)')
    parser.add_argument('--line-height', type=int, default=70,
                       help='Höhe der ausgeschnittenen Linien in Pixeln (z.B. 70)')
    parser.add_argument('--debug', help='Speichert ein annotiertes Bild mit gefundenen Zeilen', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_root = Path(args.output) if args.output else source.parent / 'prepared_lines'
    output_root.mkdir(parents=True, exist_ok=True)

    process_path(source, output_root, args.start_value, args.step, args.line_height, debug=args.debug)
    print(f"Fertig. Ergebnisse liegen in: {output_root}")


if __name__ == '__main__':
    main()
