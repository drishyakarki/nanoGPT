# nanoGPT

![](data/gpt.jpeg)

The simplest, fastest repository for training/finetuning medium-sized GPTs. This repository contains code for a Transformer-based language model, specifically the Generative Pre-trained Transformer (GPT) model. GPT is a state-of-the-art language model architecture that has achieved impressive results in various natural language processing tasks, including text generation and language understanding. It is directly inspired by Andrej Karpathy's [gpt video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1476s).

This repository contains the code that summed up from the Karpathy's makemore series, ultimately leading to this. I have implemented the code from makemore series too. You can check it out here: [Makemore series](https://github.com/drishyakarki/MakeMore) 

## Requirements

simply, install the dependencies using
```bash
pip install -r requirements.txt
```

## Training
To train the GPT model, follow these steps:

1. Install the required dependencies: `pip install -r requirements.txt`
2. Prepare your training data in a text file (`data/train.txt`).
3. Run the training script: 
```bash
cd nanoGPT
python gpt.py
```
You will also get sample outputs once the training script is completed.
