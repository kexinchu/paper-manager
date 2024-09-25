### A Survey on Multimodal Large Language Models
Institution: USTC   
Conference: IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE    
Paper Link: https://arxiv.org/pdf/2306.13549    
Source Code: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models    

##### The architecture of MLLM
- MLLM: use powerful LLMs as a brain to support different tasks, like Text, Image, Video, Audio, et.all
- Typically Architecture:
    - Pre-trained Encoders
        - taking inputs of multiple modalities, and transform these raw multi-model data into features that the LLM can understand.
        - contains multiple encoders: 
            - image raw data  -  image encoder
            - audio raw data  -  audio encoder
            ......
    - Connectors
        - for bridging the gap between different modalities, since the LLM model can only perceive text.
        - a learnable model (the Encoder and LLMs are usually use pretrained models provided by big company)
            - the Q-Former style approaches (in Figure) compress visual tokens into a smaller number of representation vectors
            - MLP Models
            - LLM 
    - Pre-trained LLMs
    - Generator

- Training:
    - A common approach is to keep pre-trained modules (e.g. visual encoders and LLMs) frozen and train a learnable interface.
    - Some method also un-freeze the encoder for training

- Inference:
    - for example: describe a picture; MLLM take a text peompt: "Please describe the picture" and a picture as input. generate a context.
    - For the image/video -> text tasks, The encoder only execute once, while the LLM use auto-regression.

<img src="./pictures/MLLM-Architeture.webp" width=400>

### Inf-MLLM: Efficient Streaming Inference of Multimodal Large Language Models on a Single GPU
Institution: Shanghai Jiao Tong University
Conference: ArXiv Sep 11 2024
Paper List: https://arxiv.org/pdf/2409.09086
Source Code:

