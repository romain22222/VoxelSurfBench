# Data preparation

`prepare_shape.py` follows the DGNS data-preparation path for visibility-supervised NeuralPull, but starts from the
original triangle mesh rather than a precomputed voxelization. It normalizes and voxelizes the mesh at an exact cubic
resolution, writes the benchmark's `.vol` and NDC-compatible `.binvox` inputs, then samples the point cloud uniformly
over the exposed voxel faces read back from the `.vol`. It also invokes DGtal `visibilityLattices` to produce the
visibility normals. The point cloud is therefore derived from the voxelization, not sampled from the GT mesh.

For example, from Windows with the DGtal binary available in WSL:

```
python dataPreparation/prepare_shape.py \
    --mesh "\\wsl.localhost\Ubuntu\home\romain22222\VolGallery\Octaflower\octa-flower17k.stl" \
    --shape-dir experiments/shapes/octaflower \
    --resolution 64 \
    --visibility-tool /home/romain22222/visibilityLattices/cmake-build-release/visibilityLattices
```

The output includes `gt/gt.ply`, `gt/octaflower.ply` (points sampled from the `.vol` surface), `gt/shape.vol`,
`gt/shape.binvox`, `gt/visnormals.xyz`, `gt/source.stl`, and a `gt/source.json` manifest. The point count, random seed,
visibility grid step and radius can be overridden; use `--overwrite` only when intentionally regenerating existing
inputs.
