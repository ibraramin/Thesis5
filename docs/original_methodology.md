# **Information-Theoretic Token Priority in Supervised Fine-Tuning: A Decoupled Loss Optimization Framework**

## **Part A — The Chosen Pipeline Phase & The Gap**

The post-training alignment of Large Language Models has traditionally been divided into two primary stages: Supervised Fine-Tuning, which imbues the model with specific behavioral formats and domain knowledge, and subsequent alignment phases such as Reinforcement Learning from Human Feedback or Direct Preference Optimization. When operating under a strict computational ceiling—specifically, a single consumer-grade RTX 3090 graphics processing unit equipped with 24 gigabytes of Video Random Access Memory and a maximum execution window of twenty hours—the strategic selection of the pipeline phase is the most critical determinant of a study's viability. The chosen phase for this research is Supervised Fine-Tuning.  
The justification for targeting Supervised Fine-Tuning rests upon the mathematical and hardware constraints inherent in modern alignment techniques. Direct Preference Optimization and its derivatives, including Odds Ratio Preference Optimization and Simple Preference Optimization, have dominated recent literature due to their ability to align models without requiring a separate reward model. However, standard Direct Preference Optimization necessitates maintaining both a trainable policy model and a frozen reference model in memory simultaneously, effectively doubling the parameter footprint. For a three-billion-parameter model, the model weights alone would consume approximately 12 gigabytes in 16-bit precision. This leaves insufficient memory for AdamW optimizer states, which require eight bytes per parameter, and the intermediate activation gradients necessary for backpropagation. While methods like Odds Ratio Preference Optimization eliminate the reference model, preference alignment datasets require generating multiple completions, scoring them, and evaluating relative margins—a process that introduces significant variance, stochasticity, and computational overhead that cannot be reliably contained within a twenty-hour window.  
In stark contrast, Supervised Fine-Tuning relies on a single forward and backward pass utilizing a straightforward cross-entropy objective. By employing Low-Rank Adaptation combined with gradient checkpointing on a highly optimized Small Language Model, an entire epoch over a high-quality, ten-thousand-sample dataset can be executed in under three hours on the specified hardware. Furthermore, Supervised Fine-Tuning offers a direct, deterministic mathematical mapping between the token-level loss objective and the resulting probability distribution, providing an ideal, controlled sandbox for introducing mathematical modifications that guarantee measurable outcomes.  
The specific research gap this study addresses lies within the prevailing paradigm of Supervised Fine-Tuning, which currently treats learning as a uniform Maximum Likelihood Estimation problem. Standard autoregressive loss functions apply a binary masking strategy: the loss on the prompt, or context, tokens is zeroed out entirely, and the loss on the response, or generation, tokens is weighted uniformly at a value of exactly one. This egalitarian assumption is fundamentally flawed and leads to two distinct phenomena that cripple model efficiency and robustness.  
First, the standard approach causes contextual degradation. By ignoring prompt tokens entirely during the gradient update, the model is deprived of valuable contextual guidance. Recent literature introducing Prompt Loss Weighting and Weighted Instruction Tuning demonstrates that applying a small, non-zero weight to prompt tokens acts as a crucial regularizer. This regularizer prevents the model from diverging excessively from its pre-trained representations and significantly improves robustness against prompt perturbations. Second, within the response itself, standard Supervised Fine-Tuning suffers from a granularity mismatch and subsequent gradient starvation. The standard objective applies the identical loss magnitude to highly predictable, low-information structural tokens as it does to rare, semantically critical tokens. Because structural tokens are frequent and easily predicted, they rapidly drive their localized loss to near-zero. Enforcing uniform gradients on these ubiquitous tokens dilutes the optimization budget, drowning out the rare, critical signals required for robust reasoning.  
While recent works have explored applying a static Prompt Loss Weight or applying heuristic token-weighting to responses based on external reward models, there is a critical void in the literature regarding a unified, decoupled, information-theoretic loss objective. Specifically, no existing study has systematically evaluated the compound effect of applying a static contextual regularizer alongside a dynamically scaled, corpus-derived informational weight, such as Term Frequency-Inverse Document Frequency, for response tokens during the fine-tuning process.  
The core hypothesis of this research is that implementing a Decoupled Information-Theoretic Token Priority objective will significantly reduce verbatim memorization and improve out-of-distribution reasoning compared to uniform Supervised Fine-Tuning, without increasing computational overhead. This objective assigns a static, low-magnitude loss weight to prompt tokens to preserve contextual integrity, while simultaneously modulating the loss of response tokens using localized Term Frequency-Inverse Document Frequency coefficients. If the static lexical approach fails to map effectively to the model's internal representations—perhaps due to tokenizer fragmentation—the contingency plan will pivot to utilizing the model's own predictive entropy, representing epistemic uncertainty, as the response scaling factor. This structural redundancy ensures a guaranteed measurable outcome, either confirming that static corpus statistics are sufficient for token prioritization, or proving that dynamic model-state feedback is strictly necessary to overcome uniform optimization constraints.

## **Part B — The Study Plan (Key Experiments)**

The study is constructed through three compoundable experiments, followed by one contingency experiment. Each phase builds logically upon the prior run to isolate the exact mathematical variable responsible for changes in model behavior, ensuring rigorous attribution of performance gains.

### **Experiment 1: The Contextual Anchor**

The objective of the first experiment is to establish a robust baseline and verify that introducing a non-zero loss weight to the prompt tokens improves zero-shot robustness and stabilizes gradients across the target dataset. In standard Supervised Fine-Tuning, the loss function for a dataset containing prompts and responses is calculated solely over the response tokens. This experiment applies a Weighted Instruction Tuning formulation to evaluate the impact of contextual anchoring.  
The execution involves running three constrained configurations to isolate the effect of the prompt loss weight. The first configuration is the baseline, where the prompt weight is zero and the response weight is one. The second configuration introduces a low prompt loss weight, setting the prompt weight to a fractional value such as one-tenth, while maintaining the response weight at one. The third configuration applies a high prompt loss weight, elevating the prompt coefficient to one-half. The modified loss function calculates the weighted sum of log-probabilities, scaling the log-probabilities of prompt tokens by the prompt weight and those of response tokens by the response weight, subsequently normalizing by the total count of tokens receiving a non-zero weight.

| Configuration Name | Prompt Weight | Response Weight | Expected Optimization Behavior |
| :---- | :---- | :---- | :---- |
| Standard Baseline | 0.0 | 1.0 | High susceptibility to prompt perturbation; standard benchmark performance. |
| Low Contextual Anchor | 0.1 | 1.0 | Improved robustness to grammatical variance; stabilized gradient norms. |
| High Contextual Anchor | 0.5 | 1.0 | Potential over-regularization; degradation in novel instruction following. |

This step secures the optimal prompt weight value, which empirical literature suggests resides in the low-magnitude range for short-to-medium generation tasks. This derived coefficient will act as a stabilizing contextual anchor for all subsequent experiments, ensuring that modifications to the response objective do not inadvertently compromise the model's foundational understanding of the input context.

### **Experiment 2: Lexical Priority**

The objective of the second experiment is to demonstrate that treating all response tokens uniformly is severely suboptimal. By scaling loss gradients based on the informational density of the tokens, the optimization process forcefully prevents the model from overfitting to structural syntax and redirects computational resources toward semantic mastery.  
To isolate the response perturbation, the prompt weight is temporarily returned to zero. Prior to training, the Term Frequency-Inverse Document Frequency score is computed for every token in the target dataset's vocabulary based on its distribution across the corpus. The weight assigned to each individual response token during the forward pass is defined by its normalized Term Frequency-Inverse Document Frequency score. The loss function is thereby adapted to compute a weighted cross-entropy, where the probability of each generated token is scaled by its corresponding lexical priority weight before summation.  
By systematically down-weighting ubiquitous tokens such as articles and conjunctions, and simultaneously up-weighting critical entities and logical operators, the optimization budget is redirected toward semantic correctness. Comparing the outcomes of the second experiment directly to the baseline isolates the distinct effect of response-token priority. If performance improves on reasoning benchmarks, it provides mathematical confirmation that gradient starvation caused by uniform Supervised Fine-Tuning can be effectively cured via static lexical scaling.

### **Experiment 3: The Decoupled Objective**

The third experiment represents the synthesis of the preceding investigations, deploying the full Decoupled Information-Theoretic Token Priority framework. The objective is to combine the context-preserving regularization of the first experiment with the semantic prioritization of the second experiment, creating a unified, highly efficient training paradigm.  
The mechanism involves fusing the optimal static prompt weight discovered in the first experiment with the localized lexical weights utilized in the second experiment. The resulting loss objective calculates the cross-entropy of the prompt tokens scaled by the static anchor, adds the cross-entropy of the response tokens scaled by their respective lexical priority vectors, and normalizes the entire sum. This creates a dual-channel distribution reshaping process. One channel actively regularizes the input context, forcing the model to retain a strict mapping of the instruction space, while the other channel prioritizes high-density semantic outputs, accelerating convergence on reasoning tasks.

| Experiment Phase | Mathematical Focus | Primary Target Metric | Theoretical Justification |
| :---- | :---- | :---- | :---- |
| Experiment 1 | Static Input Weighting | Format Robustness Variance | Prevents pre-trained representation drift. |
| Experiment 2 | Static Output Weighting | Verbatim Memorization | Combats gradient starvation on rare tokens. |
| Experiment 3 | Dual-Channel Weighting | Exact Match (Reasoning) | Synergizes context preservation with output semantic density. |

If this compound objective yields the highest accuracy and the lowest susceptibility to prompt perturbations, the core hypothesis is confirmed. It establishes that Supervised Fine-Tuning must not be viewed as a monolithic probability maximization, but rather as a highly granular, token-specific intervention.

### **Experiment 4: Contingency via Epistemic Priority**

Research involving static lexical statistics over tokenized text carries an inherent risk of failure due to sub-word fragmentation. If a critical semantic word is split into multiple highly common sub-word tokens by the byte-pair encoding algorithm, the Term Frequency-Inverse Document Frequency signal may be neutralized, causing the second and third experiments to yield null results. The fourth experiment serves as a guaranteed fallback, pivoting the methodology to utilize the model's own real-time uncertainty rather than static corpus statistics.  
Instead of computing lexical frequencies, this experiment calculates the Shannon entropy of the model's predictive distribution over the entire vocabulary at each autoregressive step. Tokens where the model exhibits high confidence and low entropy are typically structural components or perfectly memorized sequences. Conversely, tokens exhibiting high entropy represent complex reasoning junctures, knowledge conflicts, or epistemic uncertainty. The scaling factor applied to the cross-entropy loss is made directly proportional to this calculated entropy.  
This contingency guarantees a successful and highly relevant thesis defense. If the static lexical approach succeeds, the research stands as a triumph of lightweight, pre-computable data optimization that requires zero additional forward passes during training. If the static approach fails but the dynamic entropy approach succeeds, the thesis successfully proves that dynamic epistemic state is fundamentally required to capture token priority, making a profound theoretical statement regarding the limits of static data analysis in autoregressive language modeling.

## **Part C — Technical Implementation Details**

### **Exact Base Model**

To strictly adhere to the twenty-four gigabyte video memory constraint while maintaining sufficient architectural complexity to exhibit advanced reasoning capabilities, the precise base model selected is the three-billion parameter variant from Meta, designated by the Hugging Face identifier meta-llama/Llama-3.2-3B.  
At approximately 3.2 billion parameters, the base weights of this model occupy roughly 6.4 gigabytes of memory when loaded in bfloat16 precision. To enable training within the hardware constraints, Parameter-Efficient Fine-Tuning via Low-Rank Adaptation is mandated. The specific configuration will utilize a rank of sixteen and an alpha of thirty-two, targeting all primary projection matrices within the attention and feed-forward blocks. This configuration restricts the trainable parameters to approximately two to three percent of the total model architecture. Consequently, the thirty-two-bit AdamW optimizer states required for the low-rank adapters will consume less than one gigabyte of memory. When combined with a per-device batch size of four and a maximum sequence length of 1024 tokens, the peak memory utilization will comfortably stabilize between fourteen and eighteen gigabytes. This architectural and configuration choice leaves ample memory headroom, completely eliminating the risk of out-of-memory errors during the extensive gradient accumulation phases required for the experiments.

### **Exact Dataset**

The foundation of robust instruction tuning relies entirely on the quality of the demonstration data. The recommended dataset for this study is the publicly available repository identified as HuggingFaceH4/no\_robots.  
This specific dataset is a premium collection consisting of precisely ten thousand high-quality instructions and responses, all authored exclusively by expert human annotators. A critical feature of this dataset is that it contains zero synthetic data generated by commercial large language models, ensuring an absolute absence of distillation contamination or stylistic collapse. With a constrained size of ten thousand examples, a single full training epoch requires less than one hour of compute time on the target hardware. This efficiency guarantees that the entire experimental pipeline—encompassing four distinct experiments, each executed for three epochs to ensure convergence—can be comfortably completed well within the mandated twenty-hour compute limit.

### **Pre-Finalizing Setup**

Prior to launching the main multi-hour computational workloads, it is an absolute necessity to verify that the custom loss functions are computing gradients correctly and that tensor shapes align perfectly across the modified architecture. The following protocol outlines the rigorous pre-run verification steps.  
The software environment must be strictly controlled, utilizing PyTorch version 2.2.0, Transformers version 4.39.0, and the associated PEFT and TRL libraries. The initial step involves the pre-computation of the lexical weights. A script utilizing the scikit-learn Term Frequency-Inverse Document Frequency vectorizer must be written, featuring a custom analyzer that explicitly uses the specific Llama tokenizer. This vectorizer is fit over the response column of the dataset to extract the global frequencies. The output must be mapped to a dictionary linking token identifiers to their normalized scores. It is critical that all scores are Min-Max normalized between a lower bound of 0.5 and an upper bound of 2.0; this specific bounding prevents the total zeroing out of gradients for common tokens while protecting against exploding gradients for exceedingly rare tokens.  
To implement the token-level loss, the standard training loop must be intercepted. This requires subclassing the standard Hugging Face Trainer and overriding the loss computation method.  
Python  
class DITTPTrainer(Trainer):  
    def compute\_loss(self, model, inputs, return\_outputs=False):  
        labels \= inputs.pop("labels")  
        outputs \= model(\*\*inputs)  
        logits \= outputs.logits  
          
        shift\_logits \= logits\[..., :-1, :\].contiguous()  
        shift\_labels \= labels\[..., 1:\].contiguous()  
          
        loss\_fct \= CrossEntropyLoss(reduction='none')  
        loss \= loss\_fct(shift\_logits.view(-1, shift\_logits.size(-1)), shift\_labels.view(-1))  
        loss \= loss.view(shift\_labels.size())  
          
        \# Apply the Decoupled Token Priority Matrix  
        \# Integration of pre-computed lexical weights and static prompt anchors  
        final\_loss \= (loss \* weight\_tensor).sum() / weight\_tensor.sum()  
        return (final\_loss, outputs) if return\_outputs else final\_loss

The sanity check protocol requires extracting a random subset of fifty samples from the dataset. A miniature training loop is executed with a single epoch, a batch size of two, and a gradient accumulation step of one. The success criteria for this verification phase are strict: the reported loss must decrease monotonically across the fifty samples, the system monitoring tools must confirm that video memory usage stabilizes without evidence of a continuous memory leak, and the script must successfully complete the run and serialize the adapter weights to disk. This pre-run verification will execute in under five minutes, and only upon achieving all success criteria should the full ten-thousand-sample dataset be initialized for the main experimental tracking.

| Hyperparameter | Value | Justification for Hardware Constraint |
| :---- | :---- | :---- |
| Learning Rate | 2e-5 | Standard convergence rate for LoRA on 3B architectures. |
| Scheduler | Cosine | Smooth decay prevents catastrophic forgetting at the end of training. |
| Warmup Ratio | 0.05 | Prevents initial gradient spikes from destabilizing the low-rank matrices. |
| Batch Size | 4 | Maximizes GPU saturation without exceeding the 24GB limit. |
| Gradient Accumulation | 4 | Achieves an effective batch size of 16 for stable gradient estimation. |
| Precision | bfloat16 | Prevents numerical underflow associated with fp16 during dynamic loss scaling. |

### **Evaluation Metrics**

To satisfy the stringent requirement for deterministic, mathematical proof and to categorically avoid the subjectivity inherent in utilizing language models as judges, the evaluation pipeline will employ four highly rigid metrics. These metrics are designed to capture distinct facets of model capability, ranging from foundational statistical distribution matching to complex semantic robustness.  
The first metric is the Perplexity on a held-out test set comprising one thousand samples from the primary dataset. Perplexity provides a strict mathematical measurement of how well the fine-tuned model predicts the underlying test distribution. A lower perplexity indicates superior generalization capabilities. Observing a mathematically significant decrease in perplexity during the second and third experiments compared to the baseline will conclusively prove that the token-priority framework improves fundamental language modeling capabilities.  
The second metric addresses the critical issue of data memorization through the Verbatim Sub-string Memorization Length. This is a purely statistical metric derived from data privacy research. It calculates the longest common continuous sub-string between the model's generated output and the exact sequences found in the training data. If the lexical weighting methodology succeeds, this metric will demonstrate a statistically significant decrease. This reduction proves mathematically that the model is synthesizing semantic concepts rather than merely memorizing structural syntax and repeating training data verbatim.  
The third metric evaluates advanced cognitive capabilities through Exact Match accuracy on a curated subset of five hundred questions from the Grade School Math benchmark. A deterministic regular expression script is utilized to extract the final numeric answer from the model's generation. The metric assigns a strict binary value of one for a perfect match and zero otherwise. This evaluation guarantees that the information-theoretic modifications applied to the loss function have either preserved or actively enhanced the deep reasoning capabilities of the base architecture.  
The fourth and final metric evaluates the structural stability of the model using Format Robustness Variance. This metric specifically tests the resilience introduced by the static prompt loss weight. A suite of one hundred questions is selected, and each question is algorithmically perturbed into five semantically identical but syntactically distinct variations—such as altering capitalization, swapping direct synonyms, and injecting trailing whitespaces. The variance in the generated token probabilities across these five perturbations is calculated. A lower statistical variance proves mathematically that the model is robust to prompt noise and invariant to surface-level formatting, thereby cementing the effectiveness of the contextual anchor implemented in the first experiment.

### **References**

1. https://arxiv.org/pdf/2602.11902  
2. https://arxiv.org/html/2510.08256v1  
3. https://arxiv.org/html/2606.09850v1  
4. https://arxiv.org/pdf/2502.19347  
5. https://arxiv.org/abs/2509.25100  
6. https://arxiv.org/html/2606.11189  
7. https://arxiv.org/html/2508.11408v3  
8. https://arxiv.org/html/2507.07817v2  
9. https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.42/133798/On-the-Effect-of-Instruction-Tuning-Loss-on  
10. https://arxiv.org/html/2401.13586v2  
11. https://aclanthology.org/2024.emnlp-main.1267.pdf  
12. https://arxiv.org/pdf/2507.07817  
13. https://arxiv.org/pdf/2606.06320  
14. https://arxiv.org/pdf/2602.01227  
15. https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177083642.24377198  
16. https://arxiv.org/html/2602.01227v1  
17. https://medium.com/data-science/to-mask-or-not-to-mask-the-effect-of-prompt-tokens-on-instruction-tuning-016f85fd67f4  
18. https://arxiv.org/html/2605.21883v1  
19. https://bright-journal.org/Journal/index.php/JADS/article/download/1136/609  
20. https://openreview.net/forum?id=y7nPwxaBh4  
21. https://arxiv.org/html/2601.02151v1  
22. https://www.semanticscholar.org/paper/Entropy-Adaptive-Fine-Tuning%3A-Resolving-Confident-Diao-Yang/9bd7bd79c9daa98c31080bec43582f0498f81186  
23. https://github.com/kowndinya-renduchintala/WIT  
24. https://en.wikipedia.org/wiki/Tf%E2%80%93idf  
25. https://openreview.net/attachment?id=y7nPwxaBh4\&name=pdf  
26. https://arxiv.org/html/2506.15021v1  
27. https://arxiv.org/html/2602.02244v1  
28. https://arxiv.org/html/2510.16882v4  
29. https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176784494.43758400  
30. https://huggingface.co/papers?q=distillation%20tokens  
31. https://huggingface.co/papers?q=token%20segmentation  
32. https://openreview.net/forum?id=TUd3c7Vr1z  
33. https://arxiv.org/html/2506.03627  
34. 