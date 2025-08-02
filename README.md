
---

# 📢 **Multispeaker & Emotional TTS based on Tacotron 2 and WaveGlow**

This project leverages **Tacotron 2** and **WaveGlow** to generate text-to-speech (TTS) with **multiple speakers** and **emotion embeddings**. It includes scripts for **data preprocessing**, **training**, and **inference** using a multi-speaker setup for diverse emotions.


## 📝 **General Description**

This repository provides code and instructions to train a **Multispeaker & Emotional Text-to-Speech (TTS)** model based on **Tacotron 2** and **WaveGlow**. The model uses emotion embeddings and speaker features for more realistic and varied speech synthesis. This is an adaptation and personal extension of the original **Tacotron 2** and **WaveGlow** implementations by NVIDIA.

---

## 🗂️ **Code Structure**

* `tacotron2/`: Contains Tacotron 2 model architecture, data functions, and loss functions.
* `waveglow/`: Contains WaveGlow model and corresponding files.
* `common/`: Includes common layers, utilities, and audio processing functions.
* `router/`: Used to select the model during training.
* `train.py`: Main script to train the models.
* `preprocess.py`: Preprocessing script to prepare the dataset.
* `inference.ipynb`: Notebook for inference using pre-trained models.
* `configs/`: Configuration files for experiments.

---

## 🔧 **Data Preprocessing**

### **Preparing for Data Preprocessing**

1. Organize your dataset into speaker folders, each containing a `wavs` directory and a `metadata.csv` file with the format: `file_name.wav|text`.
2. Configure parameters in `configs/experiments/tacotron2.py` or `configs/experiments/waveglow.py`.
3. Run preprocessing:

```bash
python preprocess.py --exp tacotron2
```

Or for WaveGlow:

```bash
python preprocess.py --exp waveglow
```

This will process the audio files, trim silences, and generate training and validation datasets.

---

## 🎓 **Training**

### **Preparing for Training**

Configure the settings for Tacotron 2 or WaveGlow in the respective experiment file:

* Set the paths for `train.txt`, `val.txt`, pretrained models, and emotion coefficients.
* Run training for Tacotron 2:

```bash
python train.py --exp tacotron2
```

For multi-GPU:

```bash
python -m multiproc train.py --exp tacotron2
```

For WaveGlow:

```bash
python train.py --exp waveglow
```

---

## 📊 **Running TensorBoard**

Monitor the training process with TensorBoard:

1. Get the container ID:

```bash
docker ps
```

2. Start TensorBoard:

```bash
docker exec -it container_id bash
tensorboard --logdir=path_to_folder_with_logs --host=0.0.0.0
```

---

## 🤖 **Inference**

Run inference with the `inference.ipynb` notebook by providing:

1. Pre-trained Tacotron 2 and WaveGlow checkpoints.
2. Input text, speaker ID, and emotion ID.

Start a Jupyter Notebook session:

```bash
jupyter notebook --ip 0.0.0.0 --port 6006 --no-browser --allow-root
```

---

## ⚙️ **Parameters**

### **Shared Parameters**

* `epochs`: Number of epochs (Tacotron 2: 1501, WaveGlow: 1001)
* `learning-rate`: Learning rate (Tacotron 2: 1e-3, WaveGlow: 1e-4)
* `batch-size`: Batch size (Tacotron 2: 64, WaveGlow: 11)
* `grad_clip_thresh`: Gradient clipping threshold (0.1)

### **Audio/STFT Parameters**

* `sampling-rate`: 22050 Hz
* `filter-length`: 1024
* `hop-length`: 256
* `win-length`: 1024
* `mel-fmin`: 0.0 Hz
* `mel-fmax`: 8000 Hz

### **Tacotron Parameters**

* `anneal-steps`: Steps to anneal the learning rate (500/1000/1500)
* `anneal-factor`: Factor by which to anneal the learning rate (0.1)

### **WaveGlow Parameters**

* `segment-length`: 8000
* `wn_config`: Affine coupling layers configuration.

---
