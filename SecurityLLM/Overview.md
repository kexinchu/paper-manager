# LLM & Security

## 大家在研究什么
- 参考论文
```shell
Security and Privacy Challenges of Large Language Models: A Survey
Institution: Florida International University
Conference: ACM Compute Survey, Feb 2025

The Emerged Security and Privacy of LLM Agent: A Survey with Case Studies
Institution: University of Technology Sydney
```
- Categories
    - Security attack (training/fine-tuning phase)
    - Privacy attack (inference phase)
        - LLM在generate answer过程中,可能回无意中捕获或者copy训练数据中存在的敏感信息
        - 大多数现有的隐私攻击都是针对视觉模型设计的
<img src="./pictures/Overview-diff-categories.jpg" width=800>

- Details
    - Prompt Jacking
        - Prompt injection
            - 误导LLM的一种直接方法是让它们忽略之前的prompt
            - Defense: 释义(破坏指令的顺序);数据提示隔离/指令预防;基于困惑度的主动检测方法(困惑度超过阈值 => 指令受损)
        - Jailbreak attack
            - bypassing 预先定义好的约束和限制,使其违反开发人员设置的基本安全限制
            - 例如: 利用LLM的拟人化和指令遵循能力,通过要求LLM以嵌套方式响应有害查询来催眠LLM,从而绕过LLM的基本安全策略
            - Recent studies focused on jailbreaking attacks in multi-model settings.例如通过视觉对抗样本来攻击MLLM
            - Meta提出了Llama Guard方法,等
    - Adversarial Attacks (对抗性攻击)
        - 对抗性样本 $x'$ 是通过对 LLM 的训练数据进行最坏情况扰动而得到的
        - backdoor 攻击
            - 恶意用户可以在模型微调期间将少量窃取提示(后门)插入良性数据集
            - 目前backdoor攻击侧重关于简单的学习任务(如分类),在大规模LLM上尚未得到验证
            - Defence:  白盒方法: 1,微调移除backdoor,2,模型剪枝,3,通过检查激活值检测backdoor; 目前黑盒环境下的backdoor预防策略仍然缺乏
        - 数据poisoning attack
            - 训练数据,可能包含外部/未经验证的数据来源,包含被操纵的数据样本,比如将触发代码直接注入到训练数据中
            - 向RAG知识库中注入中毒样本,来攻击基于RAG的LLM agent
            - 目前很少有解决方案可以防御 LLM 中的poisoning攻击。通常,数据验证、过滤、清理和异常检测技术已被用于保护 ML 模型免受中毒攻击 (poisoning数据通常是训练数据分布中的异常值)
    
    - Gradient Leakage Attacks
        - 可以通过梯度重建隐私训练样本
        - 依赖 白盒访问模型,且只能在token级别重建训练数据
        - 在梯度中插入随机噪声、动态规划 (DP) 和同态加密。
        
    - Membership Inference Attacks (MIAs) / Inference Attack
        - 估计特定数据样本属于 LLM 代理训练数据集的概率
        - 为了缓解语言领域的 MIA,提出了几种机制,包括 dropout、自动识别、模型堆叠、DP 和对抗性正则化。

    - PII Leakage Attacks
        - PII指能够唯一识别个人的数据,如姓名,电话号码,SSN,财务记录,医疗记录等信息
        - LLM 会记忆并可能泄露单个训练sample
        - PII屏蔽 (海量数据,开销大);DP等方法


## KV Cache related Security
### I Know What You Asked, NDSS'25
```shell
I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving
Conference: NDSS'25
```
- 问题: 针对KV cache reuse下的攻击: reconstruct prompt,提取隐私信息
- 方法: PromptPeek攻击方法
- 补充知识: KV cache sharing may lead to new side channel attack vectors
    - 侧信道攻击: 本文中指观测模型的响应时间差异,来判断是否命中KV cache reuse. (无需访问模型参数,仅通过对比命中／未命中缓存的响应时延,即可以百级查询开销完成对敏感 Prompt 的逐 token 重建或识别)
- extracts one token from another user’s prompt at a time, progressively reconstructing the entire prompt by repeating the token extraction procedure.

### InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks
```shell
InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks
Institution: 
Conference: ArXiv Nov 2024
```
- 核心方法于PromptLeakage一致,都是利用可观测的侧信道信息来reconstruct用户prompt (latency:  图5显示了kv cache hit 和 miss之间的latency差异)
- 挑战: 
    - 1. search space (vocabulary size exceeding 100000 tokens)
    - 2. 可观测的latency受到网络延迟/memory scheduling lat的干扰
    - 3. Real-world constraints

### Defence
- 限制KV cache reuse 粒度; 当前attacker 每次只能确定一个token
- 引入 latency 噪声,使attacker难以区分 cache hit / miss
- 缓存隔离
- 依赖相同的tokenizer

### A First Look At Efficient And Secure On-Device LLM Inference Against KV Leakage
```shell
A First Look At Efficient And Secure On-Device LLM Inference Against KV Leakage
Conference: MobiArch ’24
Institution: Central South University & Tsinghua
Link: https://dl.acm.org/doi/pdf/10.1145/3691555.3696827?casa_token=gZoLUCD7mncAAAAA:now44Y6q0oV7fZ6vEf_3NrTaNnF1G5EkehfBUgtG_kVhWbG8wtdgQO2oZ9NzTHV9uWhsDk4OZuzu
```
- 虽然on-device LLM推理的准确性和效率已经得到了验证,但面临着安全性的严峻考验 => 移动设备的计算核心容易受到各种攻击,尤其是信息泄露。
- attacker可以利用攻击捕获的KV cache来重建整个对话,获取隐私信息
    - eg. AMD GPU上的 leftoverlocal
- 现有方法: 
    - 完全同态加密(FHE), 计算密集性太高, 导致推理性能下降(5-7个数量级),
    - 可信执行环境(TEE), 可信内存大小有限
- KV-Shield 方案
    - 思路: 1. 加密KV cache (置换); 2. 使KV cache不可见, (在TEE中处理KV pairs)
    - 置换: 对Attention中计算QKV的线性层权重, 打乱行数序 (乘上一个0-1矩阵); 得到的KV cache也是行乱序的; 恢复(逆置换): 在attention计算后, 逆置换结果, 得到正确输出
    - 逆置换在TEE中发生, 对外不可见 + size有限
    <img src="./pictures/KV-Shield_equations.jpg" width=300>

## Jailbreak Attacks
### ROBUSTKV: DEFENDING LARGE LANGUAGE MODELS AGAINST JAILBREAK ATTACKS VIA KV EVICTION
```shell
ROBUSTKV: DEFENDING LARGE LANGUAGE MODELS AGAINST JAILBREAK ATTACKS VIA KV EVICTION
link: https://arxiv.org/pdf/2410.19937
```
- jailbreak攻击: 设计jailbreak prompt,绕过开发人员设置的安全限制,将harmful query隐藏在看似无害的提示中,诱使其生成恶意响应/泄露隐私
- 核心思想: 选择性地从KV cache中移除有害查询的关键标记符,从而破坏LLM对隐藏有害请求的处理能力
    - 分析KV缓存中的注意力模式 : 监控KV缓存中的注意力得分,以确定输入中哪些部分受到模型的高度关注,哪些部分的注意力得分较低。
    - 识别潜在有害内容 : 查找相对其周围上下文注意力得分异常低的文本片段,这些片段可能是隐藏的有害查询。
    - 策略性地驱逐低重要性信息 : 从缓存中移除低重要性的信息,从而干扰模型对隐藏恶意请求的处理,同时保留正常内容。
- 逃避两难困境:  攻击者为了使越狱提示生效,需要降低有害查询在提示中的重要性,这反而使RobustKV更有效;而为了逃避RobustKV的检测,攻击者若增强有害查询的重要性,又会使越狱提示更难绕过LLM的初始安全检查

## LLM & Application Isolation
### ISOLATEGPT - NDSS'25
```shell
ISOLATEGPT: An Execution Isolation Architecture for LLM-Based Agentic Systems
Conference: NDSS'25
Institution: Washington University in St. Louis
```
- 问题: 随着大型语言模型(LLMs)如ChatGPT等被扩展为独立的计算系统(代理系统),并开始支持第三方应用程序,这些LLM应用程序及其交互使用自然语言定义,被授予访问用户数据的权限,并被允许与其他应用程序、系统和在线服务交互,由此引发了一系列安全和隐私风险。
- 目标: 隔离应用程序的执行;
- 挑战: 如何安全&无缝地让用户与在隔离环境中执行的应用程序进行交互,并为基于自然语言的交互定义安全接口。
- 设计&系统架构
    - 采用中心辐射式架构 (hub-and-spoke architecture): 
        - 一个中心节点(hub), 负责接收用户请求,并将请求路由到相应的应用程序中
        - 多个spoke,上面执行应用程序;每个spoke节点包含一个专用的LLM示例以及应用程序所需的功能描述 + API接口
    - 通信: Inter-Spoke Communication ISC
        - 核心式采用hub作为可信中介,来控制不信任spoke之间的信息流,并在信息流经过hub式进行筛选和审查,以识别和终止可能的恶意交流

### ASGARD - NDSS'25
```shell
ASGARD: Protecting On-Device Deep Neural Networks with Virtualization-Based Trusted Execution Environments
Link: https://www.ndss-symposium.org/wp-content/uploads/2025-449-paper.pdf
```
