# diffraw

WORK IN PROGRESS

This is a research idea of a differentiable version of Lightroom (postprocessing adjustments implemented as pytorch modules) and use a promptable model as a critic to optimize all the settings and postprocess the image fully automatically.

## Example usage

python demo_v2.py DSC03494.ARW --prompt "dreamy"
python demo_v2.py DSC03494.ARW --prompt "cinematic, moody, film look"
python demo_v2.py DSC03494.ARW --prompt "black and white film"
