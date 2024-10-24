## MLLM hardware-software co-design
[Reading-Notes](./Hardware-Software-Codesign.md)

## MLLM Training
[Reading-Notes](./MLLM-Train.md)

## MLLM Inference
### Inf-MLLM: Efficient Streaming Inference of Multimodal Large Language Models on a Single GPU
Institution: Shanghai Jiao Tong University
Conference: ArXiv Sep 11 2024
Paper List: https://arxiv.org/pdf/2409.09086
Source Code:

##### Key Point
- efficient streaming inference of MLLMs on a single GPU

##### Challenges & Observations
- Challenges
    - Quadratic computation complexity of attention, (to the KV cache size)
    - High memory consumption for storing KV cache.
        - some multimodel inputs may transformed into a large number of tokens. like a several-minute-long video can be converted into thousands of tokens 
    - Context length limitation of pre-trained LLMs, like 4096 for LLaMA 2
    - Long-term reasoning capacity

- Observation in MLLM (Attention Patterns)
    - recent tokens have high attention scores
    - <span style="color:red;">tokens converted from videos typically receive high attention scores</span>
        - In some VLMs, the initial tokens of the video even share over 40% of attention scores
        - But since the position of videos is unknown beforehand in the multi-round conversation, an effective method is required to identify important visual tokens dynamically.
    - high attention scores are also distributed among tokens scattered in the sequence. These tokens are attended to for dozens or hundreds of decoding steps
    - high attention scores shift forward as the multi-round inference progress
        - When a new prompt comes, the distribution of attention scores changes significantly
    
    <img src="./pictures/Inf-MLLM-observations.png" width=800>

- Motivation
    - Based on the observation 1,2,3; these tokens with high attention scores(Attention Saddle), Inf-MLLM designs an new KV cache eviction policy, delete less important token
        - Eviction & Compression
        - always keep max length of L (num of tokens)
    
    <img src="./pictures/Inf-MLLM-eviction.png">
    
    - For the character in observation 4, insert Attention Bias to dynamically update and get the "shift" character


### LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
Institution: The Ohio State University
Conference: ArXiv 26 Jun 2024
Paper List: https://arxiv.org/abs/2406.18139

##### Key Observation
- Use KV cache compression
- <span style="color:red;">the model exhibits greater attention to the textual components during the multimodal prompt encoding process.</span>

<img src="./pictures/M-LOOK-Visualization.png" width=400>



