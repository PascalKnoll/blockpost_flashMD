---
layout: distill
title: "Breaking the Femtosecond Prison: Promise and Limits of FlashMD"
description: "Can a neural network learn to respect the laws of physics without being explicitly taught? We explore this question through FlashMD, a new framework that bypasses the timestep stability limit (XX femtosecond prison) of classical integrators to predict molecular evolution directly. This post guides you from the basics of MD bottlenecks to the cutting edge of learned dynamics. We conclude with an exclusive exploratory study revealing a hidden cost to this speed: when safety nets are removed, FlashMD struggles to conserve energy, highlighting the gap between statistical accuracy and physical validity."
date: 2026-02-15
future: true
htmlwidgets: true

# anonymize when submitting
authors:
  - name: Anonymous

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-breaking-the-femtosecond-prison-promise-and-limits-of-flashmd.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Molecular Dynamics Fundamentals
    subsections: 
        - name: The Hamiltonian View of Atomic Motion
        - name: Why ab Initio Molecular Dynamics Is Expensive
  - name: "Machine-Learned Interatomic Potentials: Solving the Force Bottleneck"
    subsections:
        - name: The MLIP Breakthrough
        - name: "The Remaining Challenge: The Femtosecond Prison"
  - name: "FlashMD: Escaping the Femtosecond Prison"
    subsections:
        - name: Architecture and Design Principles
        - name: "The Non-Negotiable Constraint: E(3) Equivariance"
        - name: The Point-Edge Transformer (PET)
        - name: Long-Stride Predictions in Practice
  - name: An Exploratory Study on Failure Modes
    subsections:
      - name: "Ablation 1: Loss Weighting Between Positions and Momenta"
      - name: "Ablation 2: Mass-Scaled Loss Functions"
      - name: "The Anatomy of an Explosion"
      - name: "Summary of Findings"
  - name: Summary 
---

# Introduction
As Richard Feynman once said, everything that living things do can be understood in terms of the jigglings and wigglings of atoms<d-cite key="feynman1964relation"></d-cite> . The computational microscope used to examine this behavior is called Molecular Dynamics (MD). MD enables us to simulate atoms according to Newton's laws and observe processes such as protein folding, battery charging, and material fracture—all at the atomic scale.

However, the MD workflow has long suffered from two significant bottlenecks:

1. **Calculating the forces** between atoms at each timestep
2. **The timestep stability limit** of numerical integrators

With the advent of Machine Learning Interatomic Potentials (MLIPs), the first bottleneck has been largely addressed—replacing costly quantum mechanical calculations with learned approximations. But the second bottleneck remains: we can now compute forces cheaply and accurately, yet we are still forced to take tiny time steps ($\sim10^{-15}$ s) to maintain numerical stability<d-cite key="leach2001timestep"></d-cite>. Observing biological processes that unfold over milliseconds requires **trillions of sequential steps**—each dependent on the previous one. No amount of faster force evaluation can fix this.

FlashMD<d-cite key="bigi2025flashmd"></d-cite> tackles this second bottleneck head-on. Rather than following the traditional MD loop—calculating forces and integrating numerically with small time steps—FlashMD predicts the system's evolution directly over significantly larger time intervals, often extending one to two orders of magnitude beyond the limits of classical integrators. Across water, proteins, metals, and battery materials, it recovers the correct statistical physics at speedups of 8× to 64×.

However, standard deep learning models are fundamentally agnostic to physical laws. Classical integrators come with mathematical guarantees: they approximately conserve energy, preserve time-reversibility, and maintain long-term stability. A learned model offers none of these.

This raises a fundamental question: **Can a model learn to respect the laws of physics without being explicitly taught to do so?** That is precisely what FlashMD attempts—learning the physics and dynamical behavior of atomic systems entirely from data.

In this post, we build up from first principles—starting with the physics of molecular dynamics, through the MLIP revolution, to FlashMD's architecture and its impressive empirical results. We then go beyond the original paper: in an independent exploratory study, we stress-test FlashMD in the microcanonical (NVE) ensemble, where energy must be conserved solely by the dynamics—no thermostats, no safety nets. The results reveal that all tested models exhibit systematic energy drift and eventually diverge catastrophically—exposing the gap between statistical accuracy and physical validity.

The question is not simply whether FlashMD is fast. It is whether learned dynamics can be trusted—and our results suggest they cannot yet.

---

# Molecular Dynamics Fundamentals
Molecular Dynamics (MD) is the computational engine behind our understanding of matter in motion. At its core, MD follows a straightforward recipe: place $N$ atoms in a box, calculate the forces between them, and step forward in time using Newton's equations.

The challenge lies in the scale. As shown in the workflow below, every MD simulation is built around a **core loop** that must be executed millions to billions of times:

1. **Calculate forces** on every atom (Step 2)
2. **Integrate equations of motion** to update positions and velocities (Step 3)
3. Repeat for $10^6$ to $10^9$ steps

{% include figure.liquid path="assets/img/2026-04-27-flashmd/MD_simulation_workflow.png" class="img-fluid" caption="Figure 1: Overview of a Standard MD Simulation Pipeline. The workflow involves system setup, an iterative loop of force calculation and position integration, and a final analysis phase to compute thermodynamic properties." %}

This repetition is fundamental, not a limitation. To extract meaningful thermodynamic properties—diffusion coefficients, phase transitions, reaction rates—we need trajectories long enough to sample the system's accessible states. A single nanosecond of physical time typically requires around a million integration steps.

This raises a natural question: **which step consumes the most time?**


## The Hamiltonian View of Atomic Motion
To understand where the computational bottleneck lies, we need to look inside the simulation loop. At each step, the system evolves according to Hamilton's equations of motion—the fundamental laws governing how atoms move.

The core idea is simple: **forces come from energy gradients**. We describe the system's total energy using the Hamiltonian $H$, which splits into two parts:

$$
H(\mathbf{P}, \mathbf{Q}) = \underbrace{\sum_{i=1}^N \frac{|\mathbf{p}_i|^2}{2m_i}}_{\text{Kinetic Energy } K(\mathbf{P})} + \underbrace{V(\mathbf{Q})}_{\text{Potential Energy}}
$$

where $$\mathbf{Q} = \{\mathbf{q}_i\}_{i=1}^N$$ are atomic positions, $$\mathbf{P} = \{\mathbf{p}_i\}_{i=1}^N$$ are momenta, and $$V(\mathbf{Q})$$ is the potential energy surface (PES). 

Hamilton's equations tell us how the system evolves:

$$
\frac{d\mathbf{q}_i}{dt} = \frac{\mathbf{p}_i}{m_i} \quad , \quad \frac{d\mathbf{p}_i}{dt} = -\frac{\partial V}{\partial \mathbf{q}_i}
$$

The first equation is the easy part: the velocity is simply momentum divided by mass, which makes the kinetic energy computationally cheap. The second equation is crucial: **the force on each atom is the negative gradient of the potential energy with respect to its position**. 

In principle, if we could solve these coupled differential equations exactly, we would obtain the full trajectory of the system. In practice, however, the Hamiltonian of realistic molecular systems is far too complex to admit an analytical solution. As a result, molecular dynamics always relies on approximations.

To simulate the system, we need two things:

1. A way to compute $V(\mathbf{Q})$ and its gradient $\nabla V$.
2. A way to discretize continuous time into finite steps $\Delta t$.

**For the second requirement, we use numerical integration.**

### The Velocity Verlet Algorithm

There are many numerical integration schemes out there, but Velocity Verlet is one of the most commonly used in molecular dynamics. Thanks to its simplicity and efficiency, we’ll use it here as our example.

To solve these continuous equations on a computer, we discretize time using the Velocity Verlet integrator:

$$
\begin{aligned}
    \mathbf{p}_i &\leftarrow \mathbf{p}_i - \frac{1}{2} \nabla_{\mathbf{q}_i} V \cdot \Delta t \\
    \mathbf{q}_i &\leftarrow \mathbf{q}_i + \frac{\mathbf{p}_i}{m_i} \Delta t \\
    \mathbf{p}_i &\leftarrow \mathbf{p}_i - \frac{1}{2} \nabla_{\mathbf{q}_i} V \cdot \Delta t
\end{aligned}
$$

This algorithm is **symplectic**, meaning it preserves the geometric structure of phase space. This is crucial: symplectic integrators approximately conserve the Hamiltonian (total energy) even with finite $\Delta t$, preventing the systematic energy drift that would cause non-symplectic methods to fail over long simulations<d-cite key="hairer2006geometric"></d-cite>. But there's a catch: stability requires $\Delta t \sim 0.5\text{–}1$ femtoseconds<d-cite key="leach2001timestep"></d-cite>. Larger steps cause energy drift and numerical explosions. **This is the femtosecond prison**—the fundamental timestep barrier that limits all classical MD.

Every loop iteration requires:
- **Computing forces** via $\nabla V$ (expensive)
- **Taking a tiny timestep** (limiting)

We'll address these bottlenecks in sequence, starting with force computation.

## Why *ab Initio* Molecular Dynamics Is Expensive

The first bottleneck is evaluating $V(\mathbf{Q})$ and its gradient. Historically, this forced a painful compromise:

- ***Ab initio* methods** (e.g., Density Functional Theory) solve quantum mechanics to compute forces with chemical accuracy. But they require solving the electronic structure problem at every step, with computational cost scaling as $O(N^3)$ or worse—and large constant factors that make even small systems expensive<d-cite key="zhang2018deep"></d-cite>.

- **Classical force fields** use handcrafted functions (harmonic springs, Lennard-Jones potentials, Coulomb terms) that scale linearly with system size. But these predefined functional forms—fixed at design time—cannot adapt to bond breaking, chemical reactions, or complex polarization effects that require quantum mechanical treatment (XXX Source).

For decades, this accuracy-efficiency trade-off defined the field. Quantum accuracy meant tiny systems; large-scale simulations meant sacrificing chemistry.

**Machine Learning Interatomic Potentials (MLIPs)** changed this.

---
# Machine-Learned Interatomic Potentials: Solving the Force Bottleneck
Recall that every timestep requires evaluating $\mathbf{F}_i = -\nabla_i V(\mathbf{Q})$—the gradient of the potential energy surface. Historically, this forced a compromise: precise but computationally prohibitive Quantum Mechanics (scaling $\mathcal{O}(N^3)$), or fast but inaccurate classical force fields.

MLIPs resolve this by learning a surrogate map from atomic configurations to potential energy, reducing the complexity to $\mathcal{O}(N)$. (XXX Source)

## The MLIP Breakthrough

Instead of regressing forces directly, MLIPs parameterize the scalar potential energy $$V_\theta(\mathbf{Q})$$. Forces are obtained via automatic differentiation with respect to atomic positions:

$$\mathbf{F}_{\text{pred}} = -\frac{\partial V_\theta(\mathbf{Q})}{\partial \mathbf{Q}}$$

This approach is critical for two reasons:
1. **Physical Validity:** It guarantees a conservative force field ($\nabla \times \mathbf{F} = 0$), ensuring energy conservation.
2. **Data Efficiency:** Training on force labels provides $3N$ constraints per data point (vectors) compared to a single scalar energy constraint, significantly stabilizing the gradient descent.

### Architecture: Graph Neural Networks

Molecular systems are not grids; they are continuous 3D graphs governed by physical symmetries. Modern MLIPs (e.g., SchNet<d-cite key="schutt2017schnet"></d-cite>, NequIP<d-cite key="NequIP_Batzner2022"></d-cite>, MACE<d-cite key="MACE_ALLEGRO_leimeroth2025machine"></d-cite>) leverage Geometric Deep Learning to encode these priors:

- **Locality:** Message passing operations approximate quantum interactions, which decay with distance.
- **Symmetry:** Architectures are designed to be invariant to rotation and translation ($E(3)$ symmetry). Rotating the molecule does not change the predicted energy $V_\theta$.The resulting objective function minimizes errors in both energy and forces:

$$\mathcal{L}(\theta) = \lambda_E \|V_\theta - V_{\text{DFT}}\|^2 + \lambda_F \|\mathbf{F}_{\text{pred}} - \mathbf{F}_{\text{DFT}}\|^2$$

Training on forces directly improves generalization: force supervision provides richer gradient information and helps the model handle out-of-distribution configurations<d-cite key="chmiela2018towards"></d-cite>.


## The Remaining Challenge: The Femtosecond Prison

MLIPs successfully decoupled accuracy from computational cost. They allow us to run simulations with ab initio accuracy at millisecond inference speeds.

While MLIPs accelerate the function evaluation, they are still bound to the Velocity Verlet integrator. To maintain numerical stability, we are still forced to take femtosecond steps ($\Delta t \approx 10^{-15}s$).

Simulating long-timescale phenomena (microseconds to seconds) remains computationally intractable, not because force calculation is slow, but because the integration process is inherently serial and granular. 

To escape the femtosecond prison, we cannot simply accelerate the integrator. We must **bypass it entirely**—replacing step-by-step integration with direct trajectory prediction.

Now that we've solved force calculation, how do we escape the femtosecond prison?

---

# FlashMD: Escaping the Femtosecond Prison

MLIPs solved the force bottleneck—but they left the integration bottleneck untouched. No matter how fast we evaluate $V(\mathbf{Q})$, we are still chained to femtosecond steps by the stability requirements of Velocity Verlet.

FlashMD<d-cite key="bigi2025flashmd"></d-cite> proposes a radical alternative: **bypass both force calculation and numerical integration entirely**. Instead of learning a potential energy surface and feeding its gradients into a classical integrator, FlashMD learns the **dynamical map** itself—a neural network that directly predicts how the system evolves over a large time interval:

$$\mathcal{G}_\theta: (\mathbf{Q}_t, \mathbf{P}_t) \longrightarrow (\mathbf{Q}_{t+\Delta t}, \mathbf{P}_{t+\Delta t})$$

A single forward pass through $\mathcal{G}_\theta$ replaces hundreds of sequential Velocity Verlet steps. This enables prediction strides of **1–2 orders of magnitude** beyond the stability limit of classical integrators—turning trillions of steps into millions.

But this paradigm shift comes at a cost. Classical integrators are backed by mathematical guarantees: symplecticity, time-reversibility, and approximate energy conservation. A learned dynamical map offers none of these. Whether the resulting trajectories remain physically meaningful is an open question—one we will investigate directly in our [exploratory study](#an-exploratory-study-on-failure-modes).

First, let us understand how FlashMD is built.

XXX Here Illustration of Classical MD Loop vs. FlashMD Loop

## Architecture and Design Principles

FlashMD is designed as a modular pipeline with three stages: input embedding, a graph neural network backbone, and multi-head output prediction. This modularity is deliberate—the backbone can be swapped for any sufficiently expressive GNN, as long as it respects the symmetries of physics.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/flashmd_architecture.jpeg" class="img-fluid" caption="Figure 2: The FlashMD architecture taken from the main paper. Atomic positions and momenta are embedded into a molecular graph, processed by a GNN backbone (here: PET), and decoded into displacement and momentum predictions via separate MLP heads." %}

**Stage 1: Input Embedding.**
The current positions $\mathbf{Q}$ and momenta $\mathbf{P}$ are encoded into node and edge features of a molecular graph. A critical preprocessing step is **mass scaling**: input momenta are normalized as $\tilde{\mathbf{p}}_i = \mathbf{p}_i / \sqrt{m_i}$. Without this, heavy atoms (e.g., gold at 197 amu) would dominate the training loss by a factor of $\sim 200^2$ compared to hydrogen (1 amu), causing the model to neglect fast hydrogen vibrations entirely.

**Stage 2: GNN Backbone.**
The molecular graph is processed by a message-passing neural network that extracts local geometric features and propagates information between neighboring atoms. The default choice is the Point-Edge Transformer (PET)<d-cite key="PET_pozdnyakov2023smooth"></d-cite>, but any architecture that can operate on atomic graphs is a valid candidate—provided it handles a fundamental physical constraint we discuss next.

**Stage 3: Multi-Head Output.**
Two separate MLP heads decode the final node representations into predictions:
- **Displacement head:** predicts $\Delta \mathbf{q}_i = \mathbf{q}_i(t+\Delta t) - \mathbf{q}_i(t)$
- **Momentum head:** predicts $\mathbf{p}_i(t+\Delta t)$

At inference time, optional post-processing filters can be applied: momentum rescaling for energy conservation, thermostat/barostat coupling for ensemble control, and random rotations to enforce symmetry—a point we return to shortly.

## The Non-Negotiable Constraint: E(3) Equivariance

Regardless of which GNN backbone we choose, one physical requirement is absolute: the model must respect the symmetries of Euclidean space.

Consider a water molecule. If we rotate the entire simulation box by 90°, the physics does not change. Energies remain identical, and force vectors rotate by exactly the same 90° to follow the atoms. Formally, for any rotation matrix $\mathcal{R}$, a physically valid model $\mathcal{F}$ must satisfy:

$$\mathcal{F}(\mathcal{R} \cdot \mathbf{Q}) = \mathcal{R} \cdot \mathcal{F}(\mathbf{Q})$$

This property—**E(3) equivariance**—is trivially satisfied by classical force fields (which are derived from physics) but is *not* automatic for neural networks. A standard MLP sees atomic coordinates as plain numbers; it has no notion that a rotated molecule represents the same physical system. If equivariance is violated, the model might predict that a molecule flies apart simply because it was oriented "North" instead of "East."

Two fundamentally different strategies exist to address this:

| Strategy | Examples | Mechanism | Trade-off |
|----------|----------|-----------|-----------|
| **Hard constraints** | NequIP<d-cite key="NequIP_Batzner2022"></d-cite>, MACE<d-cite key="MACE_ALLEGRO_leimeroth2025machine"></d-cite> | Spherical harmonics baked into network layers | Exact equivariance, but computationally expensive |
| **Soft constraints** | SchNet<d-cite key="schutt2017schnet"></d-cite>, PET<d-cite key="PET_pozdnyakov2023smooth"></d-cite> | Flexible architecture + data augmentation | Fast and expressive, but equivariance is approximate |

Which strategy is preferable depends on the application. Hard-constrained models provide mathematical guarantees and are ideal when exactness matters (e.g., computing free energy differences). Soft-constrained models trade guaranteed exactness for speed and architectural flexibility—an attractive bargain when the goal is long-timescale dynamics with large strides.

FlashMD adopts the soft-constraint approach, using the Point-Edge Transformer as its default backbone.


## The Point-Edge Transformer (PET)

The Point-Edge Transformer (PET)<d-cite key="PET_pozdnyakov2023smooth"></d-cite> is FlashMD's default backbone—a rotationally unconstrained, Transformer-based graph neural network that trades exact symmetry for raw expressivity.

In a standard message-passing GNN, each atom aggregates information from its neighbors into a single summary vector—discarding which neighbor contributed what. PET instead preserves the identity of every interaction by maintaining a separate feature vector $\mathbf{f}_{ij}^{(l)}$ for each directed bond between atoms $i$ and $j$ within a cutoff radius $R_c$. These per-bond representations are the fundamental currency of the architecture.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/pet_illustration.png" class="img-fluid" caption="Figure X: The PET message-passing mechanism. For each central atom, neighbor interactions are encoded as distinct tokens and processed by Transformer self-attention, enabling many-body correlations without explicit angular descriptors." %}

**How it works, step by step:**

1. **Tokenization.** For each central atom $i$, every neighbor $j$ within $R_c$ is encoded into a distinct token. Each token fuses geometric information (the relative displacement $\mathbf{q}_j - \mathbf{q}_i$) with chemical identity (atomic species of $i$ and $j$). The result is a set of neighbor tokens—one per bond—that the model can reason over individually.

2. **Transformer self-attention.** These tokens are fed into a standard Transformer self-attention layer. This is where PET's power lies: attention allows the model to learn that the *combination* of neighbors matters, not just each neighbor in isolation. For example, it can discover that a third oxygen atom nearby weakens a hydrogen bond between two water molecules—a genuine **many-body effect** that emerges from attention weights, without any hand-crafted angular descriptor.

3. **Message update.** The Transformer outputs are reinterpreted as updated bond messages $\mathbf{f}_{ij}^{(l+1)}$, which become the inputs for the next message-passing layer. After $L$ layers, the accumulated per-atom representations are passed through a feed-forward network to predict the target property (in FlashMD's case: displacements and momenta).

**Why this matters for FlashMD.** The unconstrained design gives PET a remarkable theoretical property: even a single message-passing layer acts as a **universal approximator** with virtually unlimited body order and angular resolution<d-cite key="PET_pozdnyakov2023smooth"></d-cite>. This means PET can, in principle, represent arbitrarily complex atomic interactions—exactly the kind of expressivity needed to learn a dynamical map that replaces hundreds of Velocity Verlet steps.

The price is clear: PET imposes **no explicit rotational symmetry constraints**. It must learn equivariance from data rather than guaranteeing it by construction. This is where the runtime symmetrization described below becomes essential.

### Enforcing Symmetry at Runtime
Since the PET backbone is not intrinsically equivariant, FlashMD must learn physical symmetries from the data itself.

To achieve this, the authors employ Data Augmentation. During training, every molecular configuration is randomly rotated before being fed into the model. This forces the network to learn that the physics of a molecule is independent of its orientation in space.

While this does not provide the strict mathematical guarantees of equivariant architectures (like NequIP or MACE), it allows FlashMD to retain the raw expressivity of the PET backbone. For the bulk systems studied in the paper, this approximate equivariance proves sufficient: the model recovers correct radial distribution functions and diffusion coefficients without needing expensive geometric algebra operations.

However, as we will see in our [exploratory study](#an-exploratory-study-on-failure-modes), relying on the model to "learn" symmetry rather than enforcing it by construction may have consequences when operating in strict energy-conserving ensembles.

## Long-Stride Predictions in Practice

With the architecture in place, the central question is: **does it actually work?** Can FlashMD simulate realistic molecular dynamics with dramatically larger time steps—without losing the physics that matters?

### Benchmarking Strategy

Evaluating a learned dynamics model is subtle. Molecular dynamics is **chaotic**: two simulations starting from nearly identical states will diverge exponentially within picoseconds, even if both are perfectly correct. Comparing trajectories point-by-point is therefore meaningless.

Instead, the authors adopt a **statistical evaluation**: generate reference trajectories with a trusted conventional force field (PET-MAD), run FlashMD under identical conditions, and compare **ensemble-averaged properties**—density, radial distribution functions, phase transition temperatures, and diffusion coefficients. If these statistical fingerprints match, the dynamics are physically meaningful, regardless of whether individual trajectories agree.

FlashMD is evaluated in two configurations: a **water-specific model** (trained exclusively on liquid water, optimized for maximum accuracy) and a **universal model** (trained on chemically diverse systems, designed for broad generalization). This lets the authors probe both extremes of the accuracy–generality trade-off.

### Key Results

**Liquid Water.**
Despite its apparent simplicity, water is notoriously difficult to simulate due to its fluctuating hydrogen-bond network. At 450 K, the water-specific model reproduces radial distribution functions—the spatial arrangement of oxygen and hydrogen atoms—nearly perfectly. Temperature control via a Langevin thermostat maintains deviations under 1 K. Strides of up to **16 fs** are stable, representing a **64× speedup** over the 0.25 fs baseline typically required for rigid water models (see [Figure XX](#fig:water)).

{% include figure.liquid path="assets/img/2026-04-27-flashmd/water_experiment_flashmd.png" class="img-fluid" label="fig:water" caption="Figure X: Water Experiments FlashMD" %}

**Solvated Alanine Dipeptide.**
This small peptide in water serves as a minimal proxy for protein flexibility. The critical benchmark is the **Ramachandran plot** ([Figure XX](#fig:universal) a), which maps the accessible backbone conformations. FlashMD recovers the correct distribution of dihedral angles even at strides **32× larger** than standard MD—strong evidence that it captures meaningful conformational dynamics, not merely static snapshots.

**Aluminum Surface Pre-melting.**
Metal surfaces exhibit subtle, layer-dependent dynamics at high temperatures: surface atoms become mobile before the bulk melts. FlashMD reproduces the characteristic **anisotropic softening pattern** (different vibration amplitudes along different crystal axes) and captures the formation and migration of dynamic surface defects—all at **64 fs strides** (64× speedup) ([Figure XX](#fig:universal) b).

**Lithium Thiophosphate: Superionic Transition.**
Perhaps the most striking result involves a solid-state battery electrolyte. At elevated temperatures, lithium ions become highly mobile in a "superionic" phase—a collective transition critical for battery performance ([Figure XX](#fig:universal) c). FlashMD predicts the transition temperature at 675 K (within the expected range) and reproduces the dramatic jump in lithium conductivity, achieving an **8× speedup**. Some systematic errors appear at extreme temperatures, but the qualitative physics is captured.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/universal_experiments_flashmd.png" class="img-fluid" label="fig:universal" caption="Figure X: Universal Experiments FlashMD" %}

### What These Results Show

Across liquid, biomolecular, metallic, and ionic systems, FlashMD consistently recovers the correct statistical physics while operating at strides far beyond classical stability limits. The speedups range from 8× to 64× depending on the system, with the universal model showing remarkable transferability across chemically distinct materials.

But statistical agreement with reference simulations is not the whole story. These benchmarks were conducted with thermostats and barostats that actively regulate temperature and pressure—external controls that can mask underlying issues with the learned dynamics.

**What happens when we remove these safety nets?**

In the next section, we strip away the thermostats and run FlashMD in the most unforgiving setting: the microcanonical (NVE) ensemble, where energy must be conserved by the dynamics alone. This reveals fundamental limitations of the learned approach—and points toward what must be solved before FlashMD can be trusted as a general-purpose simulator.

# An Exploratory Study on Failure Modes

To move beyond theoretical concerns, we conduct a systematic exploratory study on a concrete system: a periodic box of 258 TIP3P water molecules simulated with OpenMM as ground truth. We then trained FlashMD models under a variety of settings and asked a blunt question: can they conserve energy in NVE rollouts—the most unforgiving test of physical correctness?

## Experimental Setup

**Ground Truth Generation.** We generated short NVE trajectories across a wide temperature range (200–700 K) using the TIP3P force field in OpenMM, saving configurations every 0.5 fs over 10 ps.

**Training Data.** From these trajectories, we build training pairs of the form $(q_t, p_t) \rightarrow (q_{t+1}, p_{t+1})$, corresponding to a prediction stride of 1 fs.

**Evaluation.** Each trained model is rolled out for 50 ps in an NVE simulation, starting from a 300 K equilibrated configuration.

Before tackle the ablation studies, we first asked a profound question: do the trained models even behave sensibly?


{% include figure.liquid path="assets/img/2026-04-27-flashmd/comparison_detailed_analysis.png" class="img-fluid" label="fig:basecomparison" caption="Figure 2: Comparison of temperature and energy stability across various model setups against the OpenMM ground truth." %}

The first results are shown in [Figure 2](#fig:basecomparison). Across different random seeds and different starting configurations, the picture is consistent. All trained models produce trajectories that look stable—no immediate explosions, no obvious numerical blow-ups.

But none of them conserve energy. Total energy drifts steadily, and temperature drops far below the target value. In other words, the models are quietly bleeding kinetic energy.

To understand where this energy is going, we looked at the momentum distributions predicted by the models and compared them to ground truth. Since kinetic energy is directly determined by momenta, any systematic bias here would immediately explain the observed cooling.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/velocity_momentum_species_grid.png" class="img-fluid" label="fig:base_mom_dist" caption="Figure 3: Momentum distribution" %}

The picture is surprisingly consistent across models ([Figure 3](#fig:base_mom_dist)). Hydrogen velocities look roughly reasonable, but oxygen momenta are systematically shifted toward lower values compared to ground truth. Since oxygen atoms carry most of the system’s kinetic energy, even a small bias here translates into a large temperature drop.

This leads us to the question: can we fix this with better loss design?

We tried two straightforward ideas:

1. Change how strongly the model is penalized for momentum errors relative to position errors.
2. Change how momentum errors are measured, so hydrogen and oxygen atoms are treated more equally.

## Ablation 1: Loss Weighting Between Positions and Momenta

The first question is simple: does it matter how much the model cares about getting momenta right vs. positions? We train models with momentum loss weights $w_p \in \{0.5, 1.0, 1.5, 2.0, 10.0\}$ while keeping the position weight fixed at $w_q = 1.0$, and let them run NVE for 10 ps.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/nve_comparison_4models_10ps_temp_energy.png" class="img-fluid" label="fig:ablation1" caption="Figure X: NVE trajectories for three momentum loss weights and OpenMM ground truth. Top: temperature evolution. Bottom: total energy. The ground truth (black) is essentially flat — none of the models come close." %}

What we found is that changing the weight mostly changes how the model fails:
- With too little momentum weight, models become unstable and quickly blow up.
- With moderate weighting, models remain numerically stable but steadily cool.
- With very large weighting, temperatures look closer to correct early on, but fluctuations become large and the system eventually starts drifting again.

We also notice something interesting at the species level: when we crank the momentum weight up to $w_p = 10$, the oxygen velocity and momentum distributions move noticeably closer to the ground truth, while hydrogen barely changes (see [Figure 6](#fig:mom10)). That hints at a mass-dependent effect in how momentum errors show up. Still, the big picture doesn’t change — energy continues to drift, which reinforces that loss reweighting alone isn’t enough to recover physically consistent dynamics.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/mom_10_velocity_momentum_species_grid.png" class="img-fluid" label="fig:mom10" caption="Figure 6: Species-wise velocity and momentum distributions under large momentum loss weighting ($w_p=10$). Oxygen improves; hydrogen does not." %}

In other words, loss weighting shifts the failure mode—from exploding, to freezing, to noisy drifting—but it never produces true energy conservation. None of the settings come close to the flat energy profile of the ground-truth simulation.

So cranking the momentum weight alone clearly isn’t the answer. We don’t actually know what the *“right”* fix is yet — but the FlashMD paper points to one reasonable idea: change how momentum errors are represented by taking atomic masses into account. Naturally, we tried that next.


## Ablation 2: Mass-Scaled Loss Functions
With a standard MSE on momenta, all atoms are treated equally — even though their masses differ drastically. Because $p_i = m_i v_i$, oxygen (mass 16) contributes about $16^2 = 256$ times more to the momentum loss than hydrogen. In principle, this should bias training toward oxygen. Yet we actually observe the opposite: oxygen velocities and momenta are poorly reproduced. This hints that the problem isn’t just weighting, but how momenta are represented in the loss in the first place.

In other words, even when oxygen dominates the loss numerically, the model still fails to capture its dynamics correctly — highlighting the need for a mass-aware reformulation. Inspired by the FlashMD paper, we address this by implementing a mass-scaled loss:

$$\mathcal{L}_{\tilde{p}} = \frac{1}{N} \sum_i \frac{\|p_i^\text{pred} - p_i^\text{true}\|^2}{m_i}, \qquad \mathcal{L}_{\Delta\tilde{q}} = \frac{1}{N} \sum_i \|\Delta q_i^\text{pred} - \Delta q_i^\text{true}\|^2 \cdot m_i$$

This is equivalent to computing MSE on the mass-scaled quantities $\tilde{p} = p/\sqrt{m}$ and $\Delta\tilde{q} = \Delta q \cdot \sqrt{m}$, ensuring that velocity errors are weighted equally regardless of atomic mass.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/mass_scaling_nve_comparison.png" class="img-fluid" label="fig:mass-scaling" caption="Figure X: NVE trajectory comparison between standard MSE and mass-scaled MSE loss. Left column shows the initial drift phase (0–10 ps), right column the full 50 ps trajectory. Both models exhibit systematic cooling relative to the ground truth." %}

The fact that mass-scaling helps *early* but not *late* suggests that it improves the model's initial momentum predictions (reducing the per-species bias), but the accumulated errors from the autoregressive rollout eventually dominate regardless. The fundamental issue is that FlashMD's single-step prediction errors, while small individually, compound systematically rather than canceling stochastically—a hallmark of non-symplectic integration.

## The Takeaway
Whether we reweight momenta, mass-scale the loss, or tune coefficients, FlashMD still exhibits systematic energy drift in NVE. The problem is not simply how much the model cares about momenta, nor how errors are weighted across atoms.

The deeper issue is structural: FlashMD performs unconstrained autoregressive prediction with no mechanism enforcing conservation laws. Small per-step biases accumulate in the same direction rather than canceling out. Over thousands of steps, that inevitably turns into macroscopic energy drift.

---

# Summary
We have seen that FlashMD introduces a fundamentally new way to think about MD simulation: bypassing both force evaluation and numerical integration by directly learning the evolution of the system. In doing so, it addresses the two central bottlenecks of classical MD—the cost of force calculations and the femtosecond timestep stability limit—and demonstrates impressive speedups.

Our independent exploratory study reveals a clear limitation. When external controls such as thermostats are removed and the model is asked to operate in the microcanonical (NVE) ensemble, FlashMD does not conserve energy. This failure persists across loss reweighting and mass-scaled formulations, indicating a structural issue rather than a tuning problem.

None of this undermines the importance of the work. FlashMD is an impressive step toward longer-stride learned dynamics. But it does make one thing clear: what is still missing are the mathematical guarantees that classical integrators provide by construction.

The question we posed in the introduction—Can a model learn to respect the laws of physics without being explicitly taught to do so?—now has a nuanced answer. For statistical properties sampled under external control, yes. For the strict, unassisted conservation of energy that defines Hamiltonian dynamics, not yet.