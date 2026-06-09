"""Command-line interface with single image and batch processing."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from analyzer import DentalColorAnalyzer


def process_single_image(analyzer: DentalColorAnalyzer, image_path: str, output_path: str, json_output: str = None) -> Dict[str, Any]:
    """Process one image and optionally save JSON."""
    print(f"Processing: {image_path}")
    result = analyzer.analyze(image_path, output_path)
    # Print human-readable summary
    print("\nResults:")
    for tooth, lab in result["tooth_lab_values"].items():
        if lab:
            print(f"  {tooth}: L*={lab['L']:.1f}, a*={lab['a']:.1f}, b*={lab['b']:.1f} (pixels={result['tooth_pixel_counts'][tooth]})")
        else:
            print(f"  {tooth}: no data")
    if result["average_lab"]:
        avg = result["average_lab"]
        print(f"\nAverage (4 incisors): L*={avg['L']:.1f}, a*={avg['a']:.1f}, b*={avg['b']:.1f}")
    print(f"Annotated image saved to: {output_path}")

    if json_output:
        with open(json_output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"JSON output saved to: {json_output}")
    return result


def process_batch(analyzer: DentalColorAnalyzer, input_dir: str, output_dir: str, json_summary: str = None) -> List[Dict[str, Any]]:
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return []

    results = []
    for img_file in image_files:
        out_file = output_path / f"{img_file.stem}_annotated{img_file.suffix}"
        try:
            res = analyzer.analyze(str(img_file), str(out_file))
            results.append(res)
            print(f"✓ {img_file.name} -> {out_file.name}")
        except Exception as e:
            print(f"✗ {img_file.name} failed: {e}")
            results.append({"image_path": str(img_file), "error": str(e)})

    if json_summary:
        with open(json_summary, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nBatch JSON summary saved to: {json_summary}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Dental Color Analyzer - Upper incisor LAB extraction")
    parser.add_argument("--input", help="Input image file or directory (for batch mode)")
    parser.add_argument("--output", help="Output image file (single mode) or directory (batch mode)")
    parser.add_argument("--white-balance", action="store_true", help="Enable white balance")
    parser.add_argument("--json", help="Save results as JSON (for single: file path; for batch: summary file path)")
    parser.add_argument("--batch", action="store_true", help="Process all images in input directory")

    args = parser.parse_args()

    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    analyzer = DentalColorAnalyzer(apply_white_balance= args.white_balance)

    if args.batch:
        input_dir = args.input
        output_dir = args.output if args.output else input_dir
        process_batch(analyzer, input_dir, output_dir, args.json)
    else:
        # Single image mode
        image_path = args.input
        output_path = args.output if args.output else "output.jpg"
        process_single_image(analyzer, image_path, output_path, args.json)


if __name__ == "__main__":
    main()
