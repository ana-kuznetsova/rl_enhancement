## RL-enhancement pipeline

### Data preprocessing

All the data preprocessing happens inside the data loaders using transform functions `make_dnn_feats` and `q_transform` defined in `preproc.py`. However, K-means clusters and Wiener filters still need to be precalculated and saved on hard drive with the following command:

```bash
python main.py --mode='data' --x_path='' --noise_path='' --out_path='' --k 5
```
To see the definitions of the arguments run:

```bash
python main.py -h
```
### Pretraining steps

For both DNN and Q-function the pretraining is discriminative: each layer of the network is trained separately and then all of the layers are pretrained at ones.