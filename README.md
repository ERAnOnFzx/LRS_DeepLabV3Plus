# LRS_DeepLabV3Plus

This is the code for LRS_DeepLabV3Plus.

## Environment setup

```bash
conda create -n MM python=3.8
conda activate MM
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install tqdm
pip install opencv-python
pip install albumentations
pip install matplotlib
```

## Datasets
Due to confidentiality agreements, the dataset we used cannot be uploaded. However, you can use other datasets by modifying the `data path`, `mask_color_map`, and `num_classes`.

## Run
To train the model, run `train.py`. You may need more than 1 GPU to run it. Or you can try setting `device_ids = [0]` to see if 1 GPU can work.

After training, the best model will be saved in `models/best_model.pth`. You may run `visualization.py` to see the performance of segmentation. **Remember to replace the data path!!!**
