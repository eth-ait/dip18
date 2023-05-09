# DIP: Data Synthesis

## Synthetic training data
This is the sample code to generate synthetic IMU data from MoCap motion sequence in SMPL format, as used by DIP. 

To run this code:
1. Download SMPL model from their [website](https://smpl.is.tue.mpg.de/) and place the folder `SMPL_python_v.1.0.0` in this path.
2. Download the MoCap sequences from [AMASS](https://amass.is.tue.mpg.de/). The code here uses a sample sequence from HumanEva.
3. Run run_demo.sh.

## Preprocessing TotalCapture
Use `read_TC_data.py` to preprocess TotalCapture data.
