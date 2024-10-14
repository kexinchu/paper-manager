# MoE
- the FFN layer in LLM be changed with multiple export models

<img src="./pictures/MoE-Architecture.png" width=400>


### Title:  Mixture-of-Experts (MoE): The Birth and Rise of Conditional Computation
Institution: Rice University  
Paper Link: https://cameronrwolfe.substack.com/p/conditional-computation-the-birth  


### Title:  AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference
Institution: Peking University & Beijing Advanced Innovation Center for Integrated Circuits  
Paper Link:https://arxiv.org/abs/2408.10284  


### Title: SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget
Conference: ACL 2024  
Institution: Shanghai Jiao Tong University  
Paper Link: https://aclanthology.org/2024.acl-long.363.pdf  

##### Key Point
- Problem
    - The high memory consumption of MoE; 
    - With limited resources, dynamically loading/unloading the parameters from/to the external memory can be a good choice.
    - Which will caose frequency parameter migration (with different requests, the selected Export models may also be different).

- Observation
    - The activation locality: 
        - the generated tokens in decode stage mostly belong to the same semantic context.
        - the AI models are usually deployed in a fixed environment and used for serving an individual user or organization

- Motivation
    - At runtime, <span style="color:red">loading the important experts into the main memory, and the unimportant experts out.</span>
    - Keep a small dynamic set of important experts, namely Virtual Exports, in the main memory for inference.
        - All original experts be stored in external memory (Actual Exports).
        - dynamic map and migration between Virtual Exports and Actual Exports; and the related parameter migration.
    - The high loading/unloading time may block computation when running MoE with layer-wise memory swapping. 
    
    - Personal understanding
        - the Virtual Experts is similar with container (with export model architectures), 
        - the Actual Experts is like a real value in the Virtual Experts 

- Designs
    - How to select the most important experts
        - based on expert importance score function to calculate the importance per Exports
        $importance(E_i, X) = \sum_{x}^{X}||x||*||G(x)_i||*||E_i||$
    
    - Offline
        - Consider the resouce and performance requirement, the author conduct fine-grained expert profiling in advance
        - gather information related to hardware memory usage, inference latency, accuracy, and I/O bandwidth
        - based on these information, to get: Profilling-guided Memory Planning

    - Online
        - Execute the model with a smaller set of experts at each time
            - Use Masked Gating to replace original gating
            
            <img src="./pictures/SwapMoE-Masked-Gating.png" width=300>

        - When the importance changed, use asynchronous loading to update each layers' expert model parameter.
        - **Will cased Accuracy reduce**.

        <img src="./pictures/SwapMoE-Overview.png" width=600>


### Title: Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference
Conference: IPDPS 2024  
Institution: The Ohio State University  
Paper Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10579139  

##### Key Point
- Problem
    - For MoE architecture, arrange export models to multiple GPU device may cause extensive all-to-all communication (2 all-to-all communication).
        - self-attention -> selected Export models
        - Export models  -> self-attention
    - Because at cluster scenario, the used self-attention and the selected Export model may located on different GPU device, or even different node.

    <img src="./pictures/ExFlow-ShowCase-All2All-Comm.png" width=400>

- Observation
    - the routing decision in previous layers will largely affect the later layer's routing decisions, and this is true for any layers in the model.

- Motivation & Designs
    - Based on the observation, We can get the correlation between different export models at different layers (the probability of being used together). Called as export affinity.
    - Let those export models with high affinity be located at same GPU device; (and if single GPU is not enough, at least let them be located at the same node to avoid acorss node communication).

    <img src="./pictures/ExFlow-Show-of-ExFlow.png" width=800>

    - Static arrangement: the export affinity is obtained in advance based on dataset Pile. The author find the affinity is insensitive to the tokens.

    <img src="./pictures/ExFlow-Insensitivity-of-Affinity.png" width=400>


##### The following paper also based on similar observation
- Title: Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference
    - Conference: ISCA 2024
    - Institution: KAIST
    - Paper Link https://arxiv.org/abs/2308.12066

    - Pre-gated MoE use the layer i-1 to predict which export layers will be used in layer i.

    <img src="./pictures/Pre-Gated-MoE.png" width=400>



# Different Architecture of MoE

### Title: Mixture of Diverse Size Experts 
Conference: ArXiv 18 Sep 2024
Institution:  Xiaomi AI Lab 
Paper Link: https://arxiv.org/pdf/2409.12210  

##### Key Point
- Problem:
    - In current MoE designs,  all experts have the same size, limiting the ability of tokens to choose the experts with the most appropriate size for generating the next token

- Motivation:
    - Design Diverse Size Exports
        - We assign experts a range of parameter sizes by setting the dimensions of the hidden layers to various lengths
        - the depth of Expert model is not changed, only dementions will be changed
    
    <img src="./pictures/MoDSE-Architecture.jpeg" width=400>

- Load Balance
    - For different model size, use expert-pair allocation.
    - The token routing selection: MoDSE exhibits an equally even distribution as the baseline. 