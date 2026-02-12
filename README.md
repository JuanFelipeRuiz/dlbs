# dlbs
Repo for dlbs-minichallenge


# Repo Structure
```text
dlbs_challenge/
├─ configs/
│  └─ baseline_no_aug.yaml
├─ dataset/
│  ├─ images/
│  │  ├─ train/
│  │  └─ val/
│  └─ labels/
├─ runs/
├─ wandb/
├─ dataset.yaml
├─ train.py
├─ validate.py
├─ infer.py
├─ old_seg_yolov11.py
├─ requirements.txt
├─ LICENSE
└─ README.md
```


## Setup Worspace:

### Enviroments

```bash
# Create a new python enviroment
python3 -m venv .venv
```

```bash
# Activate 
source .venv/bin/activate
```
Depending on if you have a cuda gpu avaialble run the upper one. Currently it supports for sure version cu126 and cu128 

```bash
pip install ".[slurm]" \
  --extra-index-url https://download.pytorch.org/whl/cu126
```
```bash
pip install ".[cpu]" \
  --index-url https://download.pytorch.org/whl/cpu
```



