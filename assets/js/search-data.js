// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-home",
    title: "home",
    section: "Navigation",
    handler: () => {
      window.location.href = "/blockpost_flashMD/";
    },
  },{id: "nav-about",
          title: "about",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blockpost_flashMD/about/";
          },
        },{id: "nav-call-for-blogposts",
          title: "call for blogposts",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blockpost_flashMD/call/";
          },
        },{id: "nav-submitting",
          title: "submitting",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blockpost_flashMD/submitting/";
          },
        },{id: "nav-reviewing",
          title: "reviewing",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blockpost_flashMD/reviewing/";
          },
        },{id: "dropdown-lt-strong-gt-2026-lt-strong-gt",
              title: "&lt;strong&gt;2026&lt;/strong&gt;",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2026/";
              },
            },{id: "dropdown-2025",
              title: "2025",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2025/";
              },
            },{id: "dropdown-2024",
              title: "2024",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2024/";
              },
            },{id: "dropdown-2023",
              title: "2023",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2023/";
              },
            },{id: "dropdown-2022",
              title: "2022",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blog-track.github.io/home/";
              },
            },{id: "post-the-trade-off-between-parallel-environments-and-steps-in-ppo",
        
          title: "The Trade-off Between Parallel Environments and Steps in PPO",
        
        description: "This blog post explores batch size in PPO-what happens when we increase the number of parallel environments versus the number of rollout steps, while keeping the total samples per update fixed. We discuss how this affects bias and variance in gradient estimation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/ppo-batch-size/";
          
        },
      },{id: "post-beyond-attention-as-a-graph",
        
          title: "Beyond Attention as a Graph",
        
        description: "We extend a graph-based perspective on attention to higher-order topological structures, exploring 2-simplicial attention and its implications for transformer depth and expressivity.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/beyond-attention-as-graph/";
          
        },
      },{id: "post-attention-sinks-from-the-graph-perspective",
        
          title: "Attention Sinks from the Graph Perspective",
        
        description: "We explore attention sinks in decoder-only transformers through the lens of message passing on graphs, revealing an intrinsic structural bias toward early tokens that may explain this phenomenon.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/attention-sinks-graph-perspective/";
          
        },
      },{id: "post-a-hitchhiker-39-s-guide-to-agent-evaluation",
        
          title: "A Hitchhiker&#39;s Guide to Agent Evaluation",
        
        description: "An introductory guide to LLM-based agents&#39; evaluation. We explore what makes agent evaluation different from traditional LLM benchmarks, how to measure success, safety, and trajectory quality, and highlight open challenges in the field.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/agent-evaluation/";
          
        },
      },{id: "post-what-can-you-do-when-you-have-zero-rewards-during-rl",
        
          title: "What Can You Do When You Have Zero Rewards During RL?",
        
        description: "If your base model has zero success rates, performing RL with outcome rewards won&#39;t do anything. What can you do then? 🤔TL;DR: simply adding easy samples to your training dataset can unlock RL training!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/zero-rewards/";
          
        },
      },{id: "post-the-witness-problem-in-multi-agent-cooperation",
        
          title: "The Witness Problem in Multi-Agent Cooperation",
        
        description: "I built cognitive modules for Concordia agents and found that agent intelligence isn&#39;t the bottleneck. Strategic cooperation fails because the observation layer can&#39;t recognize strategic behavior.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/witness-problem/";
          
        },
      },{id: "post-why-did-the-ai-model-say-that-a-methodological-practical-guide-for-explaining-multimodal-medical-decisions",
        
          title: "Why Did the AI Model Say That? A Methodological Practical Guide for Explaining...",
        
        description: "A methodological guide that summarizes key explainability techniques for understanding multimodal medical AI systems, addressing the critical challenge of understanding what these systems have learned and how they&#39;ll behave in real-world clinical settings.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/why-did-the-model-say-that/";
          
        },
      },{id: "post-why-ai-evaluations-need-error-bars",
        
          title: "Why AI Evaluations Need Error Bars",
        
        description: "AI evaluations often rely on single-run scores even though models, agents, and judges are inherently stochastic, making many reported differences unstable. This post surveys statistical tools—error bars, reliability measures, Bayesian models—that show and help manage this variance. Overall, it highlights how incorporating established statistical practices can make evaluations more trustworthy and informative.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/why-ai-evaluations-need-error-bars/";
          
        },
      },{id: "post-when-we-are-nosy",
        
          title: "When We are Nosy",
        
        description: "Machine learning systems are defined for many people, and for the design of in particular language models, calls for “social choice–based’’ methods are increasing. This seems to run counter to the practice in machine learning to “personalize’’ models. This blogpost clarifies when personalization and when social choice has its place, using the Impossibility of a Paretian Liberal by Amartya Sen.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/when-we-are-nosy/";
          
        },
      },{id: "post-uncertainty-lifecycle-in-deep-learning",
        
          title: "Uncertainty Lifecycle in Deep Learning",
        
        description: "Literature discussing uncertainty handling in deep learning considers different tasks such as uncertainty propagation, uncertainty estimation, uncertainty decomposition, uncertainty attribution and uncertainty sensitivity analysis. However, there is no proper structure explaining how these different components interact with each other at different stages of a deep learning pipeline. We propose to structure the these task by considering the flow and transformation of uncertainty through the model and position them accordingly. We call this structure as “uncertainty lifecycle”. The uncertainty lifecycle can be represented as a structured process for handling, quantifying, analyzing, and interpreting uncertainties at different stages of a deep learning pipeline.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/uncertainty-lifecycle-in-deep-learning/";
          
        },
      },{id: "post-trade-offs-in-llm-compute-for-reasoning-intensive-information-retrieval",
        
          title: "Trade-offs in LLM Compute for Reasoning-Intensive Information Retrieval",
        
        description: "The BRIGHT benchmark (ICLR 2025 Spotlight) revealed that reasoning-intensive information retrieval requires LLM-augmented pipelines, but this raises a critical resource allocation question: where should computational budget be invested for maximum effectiveness? We conduct a systematic study on BRIGHT using the Gemini 2.5 model family, evaluating trade-offs across model strength, inference-time thinking depth, and reranking depth. Our controlled experiments quantify the marginal gains of allocating compute to query expansion versus reranking, providing practical guidance for optimizing LLM-based retrieval systems on reasoning-intensive tasks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/trade-offs-in-llm-compute-for-reasoning-intensive-information-retrieval/";
          
        },
      },{id: "post-task-complexity-analysis",
        
          title: "Task Complexity Analysis",
        
        description: "We analyzed and compared 3 different sampling strategies to sample 100 questions from MATH 500 and Hendrycks&#39; 4500 (the remaining questions from Hendrycks&#39; math dataset that wasn&#39;t selected for MATH500). We ran the subsets on 13 different small (apparent difficulty (the levels assigned by Hendrycks&#39; creators) than MATH 500 results in a more stable ranking for the low performing models, but the baseline apparent difficulty split is the most stable for the top performing models.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/task-complexity-analysis/";
          
        },
      },{id: "post-don-39-t-look-up-every-token-escaping-quadratic-complexity-via-geometric-patterns-and-algorithms",
        
          title: "Don&#39;t Look Up (Every Token): Escaping Quadratic Complexity via Geometric Patterns and Algorithms",
        
        description: "Large Language Models (LLMs) have brought about a significant change in the field of artificial intelligence, where they have transitioned in scope from being specialized research tools to common resources that drive the next generation of software. With increasing model parameters and training data, LLMs demonstrate new abilities in reasoning, code generation, and solving complex problems that were once considered unattainable. However, scaling these models effectively for long-context applications uniquely poses a challenge. This is primarily due to the inherent limitations of the self-attention mechanism, which has quadratic time complexity. This quadratic bottleneck hinders applications for long documents, high-resolution images, and large codebases, among others. However, what is interesting to observe is that effectively only a few parameters are used in token computation, and most calculations are sparse. Hence, sparsity emerges as an effective solution to this problem. Rather than relying on the entire attention matrix, one can utilize an approximate or sparse version of attention to achieve almost the same results much faster. The backbone of this approach is the idea that tokens do not require the entire context; they only need local context, and thus, most of the computation carried out is wasteful. In this blog, we analyze the types of attention patterns that emerge and how to use them to our advantage for faster and efficient LLMs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/sparsity/";
          
        },
      },{id: "post-an-impossibility-trilemma-for-data-free-sampler-evaluation",
        
          title: "An Impossibility Trilemma for Data-Free Sampler Evaluation",
        
        description: "Neural samplers aim to learn to sample a target unnormalized energy potential. Sampler quality can be evaluated in a data-free manner, using only the model and the target potential, or in a data-driven manner, with additional data about the target distribution such as known modes, summary statistics, and reference MCMC samples. While data-driven eval is valuable, data-free eval has compelling conceptual advantages, raising the question of how well data-free eval could work. Here, we prove an impossibility trilemma for data-free sampler evaluation; we can only have two among i) mode-covering metric, ii) stable with finite variance, iii) universal ranking (dominance transitivity guarantee / model score does not depend on other models). This note surveys underexplored design space of data-free sampler eval metrics, and asks the community which eval properties we are willing to sacrifice in the face of the impossibility of satisfying all of them.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/sampler-eval-trilemma/";
          
        },
      },{id: "post-revisiting-the-nethack-learning-environment",
        
          title: "Revisiting The NetHack Learning Environment",
        
        description: "The NetHack Learning Environment (NLE) was proposed as a challenging benchmark to test an agents abilities to perform complex reasoning over long time horizons in a stochastic, partially-observed, procedurally generated setting. To date, no approach, including those based on reinforcement learning, using large pretrained models, using handcoded symbolic agents, imitating expert trajectories or any hybrid method has achieved significant progress towards completing the game. We take a deeper look into the mechanics and interface of the NLE and show that much of the complexity of NetHack is inaccessible due to constraints on the observation and action spaces. We propose a series of modifications and show that they meaningfully improve performance on the NLE.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/revisiting-the-nle/";
          
        },
      },{id: "post-mimicking-or-reasoning-rethinking-multi-modal-in-context-learning-in-vision-language-models",
        
          title: "Mimicking or Reasoning: Rethinking Multi-Modal In-Context Learning in Vision-Language Models",
        
        description: "Vision-language models (VLMs) are widely assumed to exhibit in-context learning (ICL), a property similar to that of their language-only counterparts. While recent work suggests VLMs can perform multimodal ICL (MM-ICL), studies show they often rely on shallow heuristics such as copying or majority voting, rather than true task understanding. We revisit this assumption by evaluating VLMs under distribution shifts, where support examples come from a dataset different from the query. Surprisingly, performance often degrades with more demonstrations, and models tend to copy answers rather than learn from them. To investigate further, we propose a new MM-ICL with reasoning pipeline that augments each demonstration with a generated rationale alongside the answer. We conduct extensive and comprehensive experiments on both perception- and reasoning-required datasets with open-source VLMs ranging from 3B to 72B and proprietary models such as Gemini 2.0 and 2.5. We conduct controlled studies varying shot count, retrieval method, rationale quality, and distribution. Our results show limited performance sensitivity across these factors, indicating that current VLMs fail to effectively utilize demonstration-level information and thus do not inherit the strong few-shot abilities of large language models (LLMs). We further conduct a mechanistic analysis showing that VLMs exhibit weak prefix matching and lack induction-head-like behavior, which potentially explains the failure of MM-ICL.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/rethinking-mmicl/";
          
        },
      },{id: "post-looking-for-tractable-definitions-to-satisfy-on-the-path-to-open-endedness",
        
          title: "Looking for Tractable Definitions to Satisfy on the Path to Open-Endedness",
        
        description: "Understanding and designing systems with open-endedness has been called the “last grand challenge of AI research” and ties to problems like accelerating scientific discovery, continual learning, and exploration. Over the past few years there has been fast progress engineering bigger interactive worlds with capacity for open-endedness and designing systems that can improve autonomously on narrow task distributions, but finding a way to quantify the gap between these and true open-ended systems remains an active challenge. In this blog post, we look at some existing open-endedness definitions and discuss what it would mean to satisfy them and construct attempts at open-endedness that do. By adjusting definitions, we get insights into different forms of partial progress towards constructing open-ended systems.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/qoe/";
          
        },
      },{id: "post-extracting-model-precision-from-20-logprobs",
        
          title: "Extracting Model Precision from 20 Logprobs",
        
        description: "We demonstrate that the internal floating-point precision of language models can be inferred from API-exposed logprobs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/precision-extraction/";
          
        },
      },{id: "post-beyond-black-box-predictions-neural-operators-as-a-bridge-to-interpretable-governing-equations-in-biology",
        
          title: "Beyond Black-Box Predictions: Neural Operators as a Bridge to Interpretable Governing Equations in...",
        
        description: "This study introduces a hybrid computational framework that integrates Neural Operators with Sparse Identification of Nonlinear Dynamics (SINDy) to recover interpretable governing equations from sparse, partially observed biological data. Using an NF-κB signaling model, we demonstrate that Fourier Neural Operators effectively reconstruct hidden state trajectories from limited measurements, serving as a resolution-independent surrogate that enables the discovery of parsimonious dynamical laws. This operator-plus-symbolic paradigm offers a scalable workflow for extracting mechanistic insights from experimental readouts where classical inference methods typically fail.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/neural-ops-in-biology/";
          
        },
      },{id: "post-loneliness-as-a-case-study-for-social-reward-misalignment",
        
          title: "Loneliness as a Case Study for Social Reward Misalignment",
        
        description: "The goal of this blogpost is to use loneliness as a clean case study of social proxy-reward misalignment in RL. We introduce a minimal homeostatic environment with loneliness drift and accumulated harm, and show that engagement-optimized agents learn short-term “social snack” policies that reduce the error signal without improving the underlying social state. This simple testbed highlights why reward inference or well-being objectives may be a better foundation than engagement proxies for socially aligned AI.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/loneliness-social-misalignment/";
          
        },
      },{id: "post-the-human-knowledge-loophole-in-the-39-bitter-lesson-39-for-llms",
        
          title: "The human knowledge loophole in the &#39;bitter lesson&#39; for LLMs",
        
        description: "Are LLMs a proof that the &#39;bitter lesson&#39; holds for NLP? Perhaps the opposite is true: they work due to the scale of human data, and not just computation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/llm-bitter-lesson/";
          
        },
      },{id: "post-charting-the-depths-interpretability-tools-to-enhance-llm-safety",
        
          title: "Charting the Depths: Interpretability Tools to Enhance LLM Safety",
        
        description: "Motivated by the increasing deployment of LLMs for safety-critical applications, we provide an accessible introduction to a practical suite of interpretability tools useful for understanding LLMs’ behavior during safety-critical decisions. Previous discussions of interpretability are often heavily focused on these methods&#39; technical aspects, rather than giving practical guidance for their immediate use; here, we provide practitioners with an overview of a range of methods for understanding LLM behavior. For each method covered, we highlight what it can and cannot tell us, and how this can help inform deployment decisions.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/interp-safety/";
          
        },
      },{id: "post-diffusion-as-infinite-hierarchical-vaes-do-diffusion-models-generalize-better-than-deep-vaes",
        
          title: "Diffusion as Infinite Hierarchical VAEs - Do Diffusion Models Generalize Better than Deep...",
        
        description: "This blogpost unifies Diffusion Models and Variational Autoencoders. We demonstrate that DPMs are mathematically equivalent to Hierarchical VAEs (HVAEs) in the limit of infinite depth. By analyzing this architectural link, we explain why diffusion models avoid the posterior collapse that plagues deep VAEs and identify the sweet spot for generalization where these models perform best.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/generalization-in-diffusion-as-infinite-hvae/";
          
        },
      },{id: "post-ready-for-general-agents-let-39-s-test-it",
        
          title: "Ready For General Agents? Let&#39;s Test It.",
        
        description: "General-purpose agents are emerging, promising seamless deployment across domains. However, we currently do not measure their adaptability to diverse, unseen settings—a core requirement for true generality. We outline the key challenges and chart a path toward a unified evaluation framework designed to guide the development of general agents.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/general-agent-evaluation/";
          
        },
      },{id: "post-divide-conquer-and-standardize-a-recursive-architecture-for-multi-agent-systems-mas",
        
          title: "Divide, Conquer, and Standardize — A Recursive Architecture for Multi-Agent Systems (MAS)",
        
        description: "The scalability and robustness of current Multi-Agent Systems (MAS) are severely constrained by the heterogeneity of communication interfaces and a reliance on fragile ad-hoc integrations. We introduce FRACTAL-MAS, a recursive architecture that standardizes orchestration through the convergence of MCP and A2A protocols, integrating a unified control loop with procedural memory grounded in Case-Based Reasoning (CBR). This design allows for continuous adaptation without fine-tuning and enables a seamless transition from rigid hierarchical structures to decentralized networks, providing a reference architecture for the robust and scalable construction of MAS.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/fractal-mas/";
          
        },
      },{id: "post-old-flashmd-bypassing-the-integrator-for-long-timescale-dynamics",
        
          title: "OLD: FlashMD - Bypassing the Integrator for Long-Timescale Dynamics",
        
        description: "In 2025, a research group of the COSMO Lab published a new framework for long-stride, universal prediction of molecular dynamics, which they call FLashMD. This new approach addresses one of the biggest challenges in computational science: the trade-off between accuracy and speed in simulating atomic-scale systems. By introducing a novel neural network architecture, FLashMD learns to predict the complex, quantum-mechanical forces governing molecular behavior, enabling simulations that are both accurate and computationally efficient. This post explores the core concepts behind FLashMD, breaks down its innovative architecture, and examines its potential to revolutionize fields from drug discovery to materials science.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/flashmd_old/";
          
        },
      },{id: "post-dynamics-of-forgetting",
        
          title: "Dynamics of Forgetting",
        
        description: "We analyze catastrophic forgetting through spectral decompositions of weights and updates, revealing when optimization refines existing circuits versus builds interfering new ones. Leveraging this, we design spectral techniques that suppress destructive update components while preserving structure.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/dynamics-of-forgetting/";
          
        },
      },{id: "post-dllm-rethinking-generation-beyond-autoregressive-models",
        
          title: "dLLM - Rethinking Generation Beyond Autoregressive Models",
        
        description: "Diffusion large language models (dLLMs) provide an alternative to autoregressive Transformers, supporting parallel token generation and flexible infilling. They excel in structured, long-horizon, or data-constrained settings, though challenges remain with output length, denoising, and blockwise generation. Hybrid approaches combining diffusion for reasoning and AR for generation show promise.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/dllm/";
          
        },
      },{id: "post-sample-blog-post",
        
          title: "Sample Blog Post",
        
        description: "Your blog post&#39;s abstract. Please add your abstract or summary here and not in the main body of your text. Do not include math/latex or hyperlinks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/distill-example/";
          
        },
      },{id: "post-hypes-and-hopes-for-causal-inference-for-brain-dynamics",
        
          title: "Hypes and Hopes for Causal Inference for Brain Dynamics",
        
        description: "We study whether modern identifiability-based nonlinear ICA methods, in particular Time-Contrastive Learning (TCL), can recover meaningful sources from realistic scalp-level brain recordings such as EEG. Using simulated data, EEG sensor data, and source-reconstructed cortical activity, we evaluate whether TCL provides representations aligned with the underlying sources.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/causal-ts/";
          
        },
      },{id: "post-square-peg-round-hole-plugging-non-sequential-data-into-sequential-language-models",
        
          title: "Square Peg, Round Hole: Plugging Non-Sequential Data into Sequential Language Models",
        
        description: "Autoregressive (AR) models are central to modern generative AI systems, yet their sequential inductive bias clashes with modalities that lack an obvious ordering, such as images, graphs, and point clouds. Despite this mismatch, AR models are widely used beyond language, owing to their scalability and controllability. This post highlights the growing set of techniques that make non-sequential data amenable to autoregressive modeling. There are two broad directions: approaches that choose or optimize a generation order for a fixed tokenization, and approaches that redesign the tokenization itself to simplify each next-token prediction step. We emphasize the tradeoffs these methods face, particularly between compression and autoregressive &amp;quot;modelability&amp;quot;. By drawing these connections, we aim to motivate future work on tokenizations tailored to the needs of autoregressive models for arbitrary datatypes.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/autoregressive-tokenization/";
          
        },
      },{id: "post-escaping-the-femtosecond-prison-promise-and-limits-of-flashmd",
        
          title: "Escaping the Femtosecond Prison: Promise and Limits of FlashMD",
        
        description: "Can a neural network learn to respect the laws of physics without being explicitly constrained? We explore this question through FlashMD, a new framework that bypasses the timestep stability limit of classical integrators to predict molecular evolution directly. We guide you from the basics of MD bottlenecks to the cutting edge of learned dynamics. We conclude with an exploratory study revealing a hidden cost to this speed: when safety nets are removed, FlashMD struggles to conserve energy, highlighting the gap between statistical accuracy and physical validity.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/escaping-the-femtosecond-prison-promise-and-limits-of-flashmd/";
          
        },
      },{id: "post-breaking-the-femtosecond-prison-promise-and-limits-of-flashmd",
        
          title: "Breaking the Femtosecond Prison: Promise and Limits of FlashMD",
        
        description: "Can a neural network learn to respect the laws of physics without being explicitly taught? We explore this question through FlashMD, a new framework that bypasses the timestep stability limit (XX femtosecond prison) of classical integrators to predict molecular evolution directly. This post guides you from the basics of MD bottlenecks to the cutting edge of learned dynamics. We conclude with an exclusive exploratory study revealing a hidden cost to this speed: when safety nets are removed, FlashMD struggles to conserve energy, highlighting the gap between statistical accuracy and physical validity.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blockpost_flashMD/blog/2026/breaking-the-femtosecond-prison-promise-and-limits-of-flashmd/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/blockpost_flashMD/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/blockpost_flashMD/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/blockpost_flashMD/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/blockpost_flashMD/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/blockpost_flashMD/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/blockpost_flashMD/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/blockpost_flashMD/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/blockpost_flashMD/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/blockpost_flashMD/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/blockpost_flashMD/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/blockpost_flashMD/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/blockpost_flashMD//feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
