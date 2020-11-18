## RL-enhancement pipeline

### Data preprocessing

1. Create noisy mel spectrograms and log-amplitude target for DNN-mapping:

```python main.py --mode='data' --nn='DNN' --x_path="/PATH/TO/CLEAN/WAV/" --out_path="DIR/TO/STORE/NOISY/SPECTROGRAMS/" --noise_path="/PATH/TO/NOISE/WAV/" --y_path="DIR/TO/STORE/TARGET/SPECS/```