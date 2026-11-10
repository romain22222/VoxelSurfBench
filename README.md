# VoxelSurfBench

[Article — coming soon](#article)

Benchmark of surface reconstruction methods from voxel data.

```
pip install -r requirements.txt
```

## Layout of a shape

```
experiments/
  shapes/<shape>/
    gt/                 inputs + ground truth
    <method>/           one folder per method
  shapes_128/<shape>/   same layout
  shapes_rot/<shape>/   same layout
```

`gt/` contains `shape.vol`, `shape.binvox`, `visnormals.xyz`, `source.json`, the sampled point cloud `<shape>.ply`,
`gt.<ext>` and (after camera setup) `camera.json`.
`base.<ext>` is the raw method output, `aligned.<ext>` the same mesh in the GT frame ([-0.5, 0.5]^3),
`results.json` the metrics (in voxel units) and run information.

## Scripts

| Script | Purpose |
|---|---|
| `add_method.py` | register a method in `methods.json` (command line, or interactive when run without parameters / with `-i`) |
| `run_shape.py --shape-dir S [--methods a,b]` | all methods on one shape |
| `run_method.py --method M [--shapes-root R]` | one method on all shapes |
| `adjust_camera.py --shape-dir S` | Polyscope with GT + `.vol`; "Save camera" writes `gt/camera.json` (view, ground plane, colors, materials...) |
| `visualize_overlays.py --shape-dir S` | Polyscope overlay of GT, `.vol`, sampled points `gt/shape.ply` and every method (`--raw` for `base.<ext>`) |
| `recompute_metrics.py --measure all` | recalculate selected C++ metrics on existing successful runs |

Common options of the runners: `--overwrite`, `--stop-on-error`, `--dry-run`, `--python`, `--surface-samples`.
All experiment data belongs below `experiments/`, which is ignored by git. `run_method.py` and
`recompute_metrics.py` search it recursively; use `--shapes-root experiments/shapes_128` to restrict the set.
`methods.json` contains the bundled methods; add more with `add_method.py` before using the runners.

## Bundled reconstruction methods

The registry includes the methods from the Reconstruction and NDC repositories, plus NeuralPull without normals (NP)
and with visibility normals (NPN). NPN is the [NeuralPull-Normals](https://github.com/romain22222/NeuralPull-Normals)
variant of the official [PyTorch NeuralPull implementation](https://github.com/mabaorui/NeuralPull-Pytorch); it adds
an orientation-invariant loss on DGtal visibility normals.
Prepare a method before running it:

```
python methods/Reproduction/build.py
python methods/NDC/build.py
python methods/NPN/build.py
```

Python methods share the project's root `.venv`, created by the first Python build. Reconstruction builds in WSL2 on
Windows. NDC compiles its extension and downloads the pretrained weights; it uses CPU PyTorch by default, or CUDA with
`--torch-backend cuda`. NP and NPN share the same CUDA-enabled build and require a CUDA-capable GPU; NPN additionally
uses DGtal visibility normals, with a default normal-loss weight of `0.1`. The NPN build clones the dedicated
NeuralPull-Normals repository. NumGrad-Pull has a reserved directory under `methods/` for its later integration.
Source checkouts and virtual environments are ignored by Git.

## Preparing a shape

`dataPreparation/prepare_shape.py` builds the benchmark inputs from an original triangle mesh (not an existing
voxelization): it normalizes and samples the mesh, creates `.vol` and `.binvox` grids at the requested resolution, and
computes DGtal visibility normals. For example, on Windows with the visibility tool available in WSL:

```
python dataPreparation/prepare_shape.py \
    --mesh "\\wsl.localhost\Ubuntu\home\romain22222\VolGallery\Octaflower\octa-flower17k.stl" \
    --shape-dir experiments/shapes/octaflower \
    --resolution 64 \
    --visibility-tool /home/romain22222/visibilityLattices/cmake-build-release/visibilityLattices
```

The resulting `gt/` folder contains the normalized reference mesh, sampled point cloud, 64³ volume and BINVOX inputs,
visibility normals, the original source mesh and a preparation manifest.

## Registering a method

A method is any command that reads the inputs of `gt/` and writes a mesh to `$OUTPUT_MESH` (`base.<ext>`).
Its raw output is converted to the GT frame from the `alignment` entry (`frame`, `swap_xz`, `scale_divisor`,
`offset_voxels`, `offset_voxels_by_grid`), see `alignment.py`. Available placeholders are listed in `pipeline.py`.
`methods/command_runner.py` centralizes OS-aware command execution; pass an executable and argument list instead of
assembling a shell command string (the Reconstruction adapter uses a quoted Bash script only for WSL).

```
python add_method.py --name mc --path '$ROOT/adapters/run_mc.py' --args '--vol $VOL --output $OUTPUT_MESH' \
    --output-ext obj --swap-xz --offset-voxels 1.0
```

## Supporting a new 3D format

Add `3dObjectHandlers/<format>.py` with a subclass of `Object3D` (template: `3dObjectHandlers/3dObject.py`)
implementing `load`, `save` and `register_polyscope`. It is discovered automatically.
Mesh handlers currently cover OBJ, OFF and PLY; the VOL handler reads, writes and visualizes voxel grids.

## Metrics

Every benchmark metric is computed by a C++ executable under `metrics/`: `objCompare.cpp` handles sampled surface
distances and normal angles, `objTopology.cpp` topology and area/volume, and `objBbox.cpp` bounding-box measures.
Python only handles mesh I/O, invokes the executables and stores their JSON results. To build:

```
cmake -S metrics -B metrics/build
cmake --build metrics/build --config Release
```

The recompute command accepts the same selection and execution controls as the original project: `--measure`
(`all`, `trim90`, `normal_angles`, `new`, or a metric key), `--shapes-root`, `--methods`, `--exclude-methods`,
`--surface-samples`, `--workers`, `--since`, and `--dry-run`.

## Article

The paper is not yet published. A public article link will be added here when it is available.
