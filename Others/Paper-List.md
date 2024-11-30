### Title: OptLLM: Optimal Assignment of Queries to Large Language Models
Institution: The University of Newcastle & Chongqing University    
Conference: ArXiv May 24 2024    
Paper Link: https://arxiv.org/html/2405.15130v1    

##### Key Point
- selecting an appropriate model for each query when leveraging LLMs in order to achieve a trade-off between cost and performance expectations.
- Problems
    - Different LLMs achieve different performance at different costs. A challenge for users lies in choosing the LLMs that best fit their needs, balancing cost and performance
- We treat the problem of assigning queries to LLMs as a multi-objective optimization problem
    - propose a multi-label classification model to predict 将request分配给哪一个model
    - Optimization: based on accuracy + costs to optimize the model.

