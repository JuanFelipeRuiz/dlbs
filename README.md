# dlbs
Repo for dlbs-minichallenge

# Setup instructions:

1. save train- / val / test images to:
    - dataset/images/train
    - dataset/images/val
    - dataset/images/test

2. save train- / val- / test annotations to:
    - dataset/annotations/train
    - dataset/annotations/val
    - dataset/annotations/test

3. define dataset.yaml with configuration for dataset infos
   - example: 
        - train: dataset/images/train (path to train images)
        - val: dataset/images/val (path to val images)
        - test: dataset/images/test (path to test images, optional)
        - nc: 3 (number of classes)
        - names: ['BARGELLO', 'VERSACE', 'ZARA']
        - 
4. install requirements:
    - install your pytorch version
    - pip install -r requirements.txt

5. run train.py (usage see train.py --help for options)

6. run infer.py (usage see infer.py --help for options)

7. run validate.py (usage see validate.py --help for options)



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



