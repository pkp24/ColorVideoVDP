# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ColorVideoVDP is a full-reference visual quality metric implemented in PyTorch that predicts perceptual differences between pairs of images or videos. It is unique as the first color-aware metric accounting for spatial and temporal aspects of vision, using the castleCSF contrast sensitivity model. The metric outputs quality scores in JOD (Just-Objectionable-Difference) units, where 10 represents perfect quality and lower values indicate increasing distortion.

**Key features:**
- Models chromatic and achromatic contrast sensitivity
- Handles SDR and HDR content with colorimetric calibration
- CUDA-accelerated via PyTorch (CPU fallback available)
- Outputs quality scores, heatmaps, and distograms

## Installation and Setup

### First-Time Setup
```bash
# Create conda environment
conda create -n cvvdp python=3.13
conda activate cvvdp

# Install dependencies
conda install ffmpeg conda-forge::freeimage
conda install nvidia/label/cuda-12.9.1::cuda-toolkit  # For CUDA support

# Install from source (editable)
cd ColorVideoVDP
pip install -e .

# Or install from PyPI
pip install cvvdp
```

### For OpenEXR Support (HDR)
```bash
conda install -c conda-forge openexr-python
pip install pyexr

# On Linux, may need to update library path in ~/.bashrc:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/conda/miniconda3/lib
```

## Common Commands

### Basic Usage
```bash
# Compare single test vs reference (images or videos)
cvvdp --test test_file --ref ref_file --display standard_fhd

# Compare multiple test files against same reference
cvvdp --test test_*.mp4 --ref reference.mp4 --display standard_4k --result results.csv

# Generate heatmap and distogram visualizations
cvvdp --test test.mp4 --ref ref.mp4 --display standard_fhd --heatmap supra-threshold --distogram

# Compare HDR content
cvvdp --test test_hdr.mp4 --ref ref_hdr.mp4 --display standard_hdr_pq

# Process video frames as images with custom frame rate
cvvdp --test test_frame_%05d.png --ref ref_frame_%05d.png --display standard_4k --fps 30

# Select frame range (Matlab notation: first:step:last)
cvvdp --test test_frame_%05d.png --ref ref_frame_%05d.png --display standard_4k --fps 30 --frames 10:2:50

# Run on CPU instead of GPU
cvvdp --test test.mp4 --ref ref.mp4 --display standard_4k --device cpu

# Interactive mode (avoid repeated PyTorch initialization)
cvvdp --interactive
# Then enter one complete command per line (without 'cvvdp' prefix)
```

### Advanced Options
```bash
# Use custom display model
cvvdp --test test.mp4 --ref ref.mp4 --config-paths ./my_displays --display my_display

# List available displays
cvvdp --display ?

# Run alternative metrics
cvvdp --test test.mp4 --ref ref.mp4 --display standard_4k --metric cvvdp-ml-saliency
cvvdp --test test.mp4 --ref ref.mp4 --display standard_4k --metric pu-psnr-rgb2020

# Preview display model output (debugging)
cvvdp --test test.mp4 --ref ref.mp4 --display standard_4k --metric dm-preview --output-dir ./debug

# Dump intermediate processing stages
cvvdp --test test.mp4 --ref ref.mp4 --display standard_4k --dump-channels temporal lpyr difference

# Limit GPU memory usage
cvvdp --test test.mp4 --ref ref.mp4 --display standard_4k --gpu-mem 4.0

# Compare videos with different frame rates
cvvdp --test test_30fps.mp4 --ref ref_60fps.mp4 --display standard_4k --temp-resample
```

### Development Commands
```bash
# Run with verbose logging
cvvdp --test test.mp4 --ref ref.mp4 --display standard_4k --verbose

# Extract features for metric retraining
cvvdp --test test.mp4 --ref ref.mp4 --display standard_4k --features

# Run examples from project root
python examples/ex_simple_image.py
python examples/ex_simple_video.py
```

## Architecture Overview

### Core Components

**Metric Classes** (in `pycvvdp/`):
- `vq_metric.py` - Base class for all metrics with `predict()` and `predict_video_source()` methods
- `cvvdp_metric.py` - Main ColorVideoVDP implementation (class `cvvdp`)
- `cvvdp_ml_metric.py` - ML-enhanced variants (`cvvdp_ml_saliency`, `cvvdp_ml_transformer`)
- `psnr_metric.py` - PSNR variants (PU-PSNR, RGB PSNR)
- `ssim_metric.py` - SSIM implementation

**Display Models** (`display_model.py`):
- `vvdp_display_photometry` - Base class for display color/brightness handling
- `vvdp_display_photo_gog` - Gain-offset-gamma model for SDR displays
- `vvdp_display_photo_absolute` - Absolute luminance model for HDR displays
- `vvdp_display_geometry` - Display size, resolution, and viewing distance

**Video I/O**:
- `video_source.py` - Base class `video_source` with frame iteration
- `video_source_file.py` - Loads images/videos via FFmpeg or imageio
- `video_source_yuv.py` - Raw YUV file support
- `video_writer.py` - Writes output videos/images

**Processing Pipeline**:
- `csf.py` - `castleCSF` class implementing contrast sensitivity function
- `lpyr_dec.py` - Laplacian pyramid decomposition and contrast encoding
- `interp.py` - 1D/3D interpolation utilities
- `visualize_diff_map.py` - Heatmap generation
- `dump_channels.py` - Exports intermediate processing stages

**Entry Points**:
- `run_cvvdp.py` - Command-line interface (`main()` function creates `cvvdp` binary)
- `__init__.py` - Public API exports

### Configuration Files

Located in `pycvvdp/vvdp_data/`:
- `display_models.json` - Display specifications (standard_4k, standard_fhd, standard_hdr_pq, etc.)
- `color_spaces.json` - Color space definitions
- `cvvdp_parameters.json` - Metric calibration parameters
- `csf_lut_*.json` - Contrast sensitivity lookup tables

Custom configs can be provided via `--config-paths` (must start with same base name, e.g., `display_models_custom.json`).

### Dimension Ordering

The `predict()` method accepts numpy arrays or PyTorch tensors with flexible dimension ordering via `dim_order` parameter:
- `B` - Batch
- `C` - Color channel
- `F` - Frame
- `H` - Height
- `W` - Width

**Examples:**
- `"HWC"` - Single color image (height, width, 3 channels)
- `"FCHW"` - Video with frames-first ordering
- `"BCFHW"` - Batched video (default, most efficient)

## Python API Usage

### Basic Image Comparison
```python
import pycvvdp

I_ref = pycvvdp.load_image_as_array('reference.png')
I_test = pycvvdp.load_image_as_array('test.png')

metric = pycvvdp.cvvdp(display_name='standard_4k', heatmap='threshold')
JOD, stats = metric.predict(I_test, I_ref, dim_order="HWC")

print(f"Quality: {JOD:.3f} JOD")
# Access heatmap from stats['heatmap'] if requested
```

### Video Comparison
```python
import numpy as np
import pycvvdp

# Assume V_test and V_ref are numpy arrays with shape (H, W, C, F)
metric = pycvvdp.cvvdp(display_name='standard_4k')
JOD, stats = metric.predict(V_test, V_ref, dim_order="HWCF", frames_per_second=30)
```

### Using as Loss Function
```python
metric = pycvvdp.cvvdp(display_name='standard_4k')
metric.train(True)  # Set to training mode

# In training loop:
loss = metric.loss(output_images, target_images, dim_order="BCHW")
loss.backward()
```

**Caveats for optimization:**
- ColorVideoVDP disrupts loss landscape convexity - use with L1/L2 losses
- Better suited for lower-dimensional problems
- Consider using only in later training stages after L1/L2 convergence

### Custom Display Models
```python
from pycvvdp import vvdp_display_photo_gog, vvdp_display_geometry

# Create custom display geometry
geom = vvdp_display_geometry(
    resolution=[3840, 2160],
    viewing_distance_meters=1.5,
    diagonal_size_inches=32
)

# Create custom photometry
photo = vvdp_display_photo_gog(
    contrast=1000,
    max_luminance=300,
    E_ambient=100,
    k_refl=0.01
)

metric = pycvvdp.cvvdp(display_photometry=photo, display_geometry=geom)
```

## Display Specification Requirements

Unlike most metrics, ColorVideoVDP requires physical display specifications:
- **Resolution** - Native display resolution (e.g., 3840x2160)
- **Viewing distance** - Meters or as diagonal_size_inches with automatic calculation
- **Max luminance** - Peak brightness in cd/m² (nits)
- **Contrast** - Static contrast ratio
- **Ambient light** - E_ambient (lux) and k_refl (reflectance coefficient)

Use standard displays (standard_4k, standard_fhd, standard_hdr_pq) for reproducible results across studies.

**Image vs Display Resolution:**
- If image is smaller than display, metric assumes it occupies central portion
- If image is larger, metric assumes a larger display
- Use `--full-screen-resize {bilinear,bicubic,lanczos}` to force full-screen scaling

**Online calculator** for display parameters: https://www.cl.cam.ac.uk/research/rainbow/projects/display_calc/

## Interpreting JOD Scores

- **10 JOD** - Perfect quality (no visible difference)
- **9 JOD** - Very high quality (minor artifacts)
- **7-8 JOD** - Good quality (visible but not objectionable)
- **5-6 JOD** - Fair quality (noticeable degradation)
- **<5 JOD** - Poor quality (significant distortion)

**Preference prediction:** Difference in JOD = probability of preferring higher-scored option:
- **1 JOD difference** → 75% prefer higher-scored version
- **2 JOD difference** → 90% prefer higher-scored version
- **3 JOD difference** → 96% prefer higher-scored version

## Visualization Outputs

### Heatmaps (`--heatmap`)
- `threshold` - Green to red (0-1 JOD) for small differences
- `supra-threshold` - Blue to yellow (0-3 JOD) for large differences
- `raw` - Grayscale (0-10 JOD range)

### Distogram (`--distogram [max_jod]`)
Visualizes differences per visual channel (achromatic, red-green, blue-yellow) and per frame over time. Optional parameter sets maximum JOD value for visualization (default: 10).

### Intermediate Channels (`--dump-channels`)
Export intermediate processing stages:
- `temporal` - After temporal filtering
- `lpyr` - Laplacian pyramid decomposition
- `difference` - Per-band difference signals

## Matlab Interface

```matlab
% Create wrapper with conda environment name
v = cvvdp('cvvdp');

% Load images
img_ref = imread('reference.png');
img_test = imnoise(img_ref, 'gaussian', 0, 0.001);

% Compare
[jod, heatmap] = v.cmp(img_test, img_ref, 'standard_fhd', 'heatmap', 'threshold');
```

See `matlab/cvvdp.m` for full wrapper implementation.

## Available Metrics

Select with `--metric` (or `-m`):
- **cvvdp** - Original ColorVideoVDP (default, recommended)
- **cvvdp-ml-saliency** - ML-based variant with saliency model (experimental)
- **cvvdp-ml-transformer** - Transformer-based variant (experimental, better than saliency)
- **psnr-rgb** - PSNR on native RGB (uses PU21 for HDR)
- **pu-psnr-rgb2020** - PSNR on BT.2020 RGB with PU21 encoding
- **pu-psnr-y** - PSNR on PU21-encoded luminance
- **dm-preview** - Debug display model (outputs HDR video or OpenEXR frames)

ML variants are calibrated for streaming distortions (bitrate/resolution) but may not generalize well. Not recommended for optimization.

## Reporting Results

When publishing results, include the metric info string (printed by default):
```
"ColorVideoVDP v0.5.4, 75.4 [pix/deg], Lpeak=200, Lblack=0.5979 [cd/m^2], (standard_4k)"
```

This ensures reproducibility with display specifications.

## Development Notes

### Project Structure
```
ColorVideoVDP/
├── pycvvdp/                 # Main package
│   ├── vvdp_data/           # Config files (JSON)
│   ├── csf_cache/           # Precomputed CSF lookup tables
│   ├── third_party/         # External dependencies (ssim, cpuinfo)
│   └── *.py                 # Core implementation
├── examples/                # Python usage examples
├── matlab/                  # Matlab wrapper
├── calibration/             # Training scripts (extract_features.py, train.py)
├── example_media/           # Test images/videos
└── pyproject.toml           # Package configuration
```

### Adding Custom Metrics

1. Create new metric class inheriting from `vq_metric`:
```python
from pycvvdp.vq_metric import vq_metric, register_metric

class my_metric(vq_metric):
    def predict_video_source(self, vid_source, frame_padding="replicate"):
        # Implementation
        return quality_score, stats_dict

    def quality_unit(self):
        return "dB"  # or "JOD", etc.

register_metric(my_metric)
```

2. Import in `__init__.py` or `run_cvvdp.py` to make available via CLI

### GPU Memory Management

The metric automatically estimates how many frames can be processed simultaneously based on available GPU memory. If encountering OOM errors:
- Use `--gpu-mem <GB>` to manually limit memory usage
- Process fewer frames with `--nframes`
- Switch to CPU with `--device cpu` (slower)

### Temporal Processing

ColorVideoVDP requires ≥250ms of video for temporal filtering. First frames are padded using:
- `replicate` - Repeat first frame (default)
- `circular` - Wrap around to last frames
- `pingpong` - Mirror first frames

Configure via `--temp-padding` or `frame_padding` parameter in Python API.

## Troubleshooting

**Import errors after installation:**
- Ensure conda environment is activated
- Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

**Slow processing on GPU:**
- Check GPU is being used: run with `--verbose` and verify device output
- For Mac: ensure torch>=2.1.0 for MPS support, use `--device mps`

**Incorrect display model:**
- Run with `--display ?` to list available models
- Check loaded config with `--verbose`
- Verify custom config file naming: must start with `display_models_`, `color_spaces_`, or `cvvdp_parameters_`

**Video loading failures:**
- Ensure FFmpeg is installed: `ffmpeg -version`
- For frame sequences, always specify `--fps`
- Check video metadata: `ffprobe video.mp4`

**HDR content issues:**
- Use display with correct EOTF (e.g., `standard_hdr_pq` for PQ-encoded video)
- For OpenEXR, ensure absolute linear values (1000 = 1000 cd/m²)
- Verify `colorspace` field in display model matches content
