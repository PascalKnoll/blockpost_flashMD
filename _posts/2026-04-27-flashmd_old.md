---
layout: distill
title: "OLD: FlashMD - Bypassing the Integrator for Long-Timescale Dynamics"
description: "In 2025, a research group of the COSMO Lab published a new framework for long-stride, universal prediction of molecular dynamics, which they call FLashMD. This new approach addresses one of the biggest challenges in computational science: the trade-off between accuracy and speed in simulating atomic-scale systems. By introducing a novel neural network architecture, FLashMD learns to predict the complex, quantum-mechanical forces governing molecular behavior, enabling simulations that are both accurate and computationally efficient. This post explores the core concepts behind FLashMD, breaks down its innovative architecture, and examines its potential to revolutionize fields from drug discovery to materials science."
date: 2026-04-27
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
bibliography: 2026-04-27-flashmd.bib

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
        - name: Why Faster Forces Are Still Not Enough
  - name: "FlashMD: Escaping the Femtosecond Time-Step"
    subsections:
        - name: A Shift from Forces to Trajectories
        - name: Architecture and Design Principles of FlashMD
        - name: Long-Stride Predictions in Practice
  - name: Limitations and Open Challenges
    subsections:
      - name: "Learning Dynamics Without Physical Guarantees"
      - name: An Exploratory Study on Failure Modes
      - name: "Future Direction: Active Learning Strategies"
  - name: Conclusion 
---

# Introduction
Molecular Dynamics (MD) is often called the "computational microscope" of modern science. By simulating atoms obeying Newton's laws, we can watch proteins fold, batteries charge, and materials fracture—all at the atomic scale. If we can simulate the atoms, we can predict the material.

But there's a brutal trade-off: **accuracy versus time**.

Machine Learning Interatomic Potentials (MLIPs) recently solved the first bottleneck—calculating forces—by replacing expensive quantum calculations with learned approximations. But a second, more stubborn barrier remains: the **femtosecond prison**. Traditional integrators must take tiny time steps ($\sim10^{-15}$ s) to remain stable<d-cite key="leach2001timestep"></d-cite>. To observe a biological process lasting milliseconds requires **trillions of sequential steps**—each depending on the last.

FlashMD shatters this chain. By directly predicting the system's state over time steps $1-2$ orders of magnitude higher than the stability limit of traditional integrators, it shifts the paradigm from **simulation to emulation**. Instead of numerically integrating forces at every femtosecond, FlashMD learns to leap forward in time, promising to collapse simulations that once took weeks into hours.

**But here's the catch:** Can a learned emulator remain physically stable? 

Traditional integrators come with mathematical guarantees—energy conservation, time-reversibility, symplectic structure. FlashMD trades these guarantees for speed. It learns dynamics purely from data. In this post, we'll explore FlashMD's architecture and ambitions—and then critically examine whether learned emulation can escape the femtosecond prison without losing physical validity. Through an exploratory study, we'll see where it breaks down and what that reveals about the future of learned simulators.



<!-- Molecular Dynamics (MD) is often described as the "computational microscope" of modern science. By solving Newton’s equations of motion for atomistic systems, it provides our only window into the dynamic behavior of matter—from the folding of proteins to the diffusion of ions in a battery. If we can simulate the atoms, we can predict the macroscopic properties of the material.

However, a frustrating trade-off has always plagued these simulations: the battle between **accuracy** and **time**.

While the introduction of Machine Learning Interatomic Potentials (MLIPs) successfully bridged the gap between quantum accuracy and classical speed for calculating forces, a second, more stubborn bottleneck remains. Standard simulations are shackled by the stability of their numerical integrators, which require time steps on the order of femtoseconds ($10^{-15}$ s)<d-cite key="leach2001timestep"></d-cite> . To observe biological processes that occur over milliseconds ($10^{-3}$ s), a simulation must execute trillions of sequential steps.

FlashMD is a new machine learning framework that attempts to shatter this barrier. Instead of painstakingly integrating the forces on every atom at every femtosecond, FlashMD uses a deep-learning architecture to "leap" forward in time. By directly predicting the system's state over time steps $1-2$ magnitudes higher than the stability limit of traditional integrators, it aims to shift the paradigm from simulation to emulation.

In this post, we will first explore the fundamental bottlenecks that made FlashMD necessary. We will then dive into its architecture, and finally, offer a critical analysis of its physical validity—specifically regarding the crucial issue of energy conservation. -->

---

# Molecular Dynamics Fundamentals
Molecular Dynamics (MD) is the computational engine behind our understanding of matter in motion. At its core, MD follows a straightforward recipe: place $N$ atoms in a box, calculate the forces between them, and step forward in time using Newton's equations.

The challenge lies in the scale. As shown in the workflow below, every MD simulation is built around a **core loop** that must be executed millions to billions of times:

1. **Calculate forces** on every atom (Step 2)
2. **Integrate equations of motion** to update positions and velocities (Step 3)
3. Repeat for $10^6$ to $10^9$ steps

[INSERT FIGURE HERE]

This repetition is fundamental, not a limitation. To extract meaningful thermodynamic properties—diffusion coefficients, phase transitions, reaction rates—we need trajectories long enough to sample the system's accessible states. A single nanosecond of physical time typically requires around a million integration steps.

This raises a natural question: **which step consumes the most time?**

<!-- Molecular Dynamics (MD) is often described as a "computational microscope." It allows researchers to observe how atoms and molecules interact over time, providing insights into dynamic behaviors that static images cannot capture.

In many ways, an MD simulation parallels a real laboratory experiment. The workflow typically follows three stages:

1. **Preparation (Equilibration):** We select a model system of $N$ particles and solve Newton's equations until the system settles into a stable state.
2. **Measurement (Production Run):** We evolve the system further to measure macroscopic properties—such as temperature, pressure, or diffusion coefficients
3. **Analysis:** Since instantaneous measurements are noisy, we average these properties over time to obtain statistically significant results. -->
   
  
XXX Illustration of a MD Simulation witht the steps

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

The second equation is crucial: **the force on each atom is the negative gradient of the potential energy**. To simulate the system, we need two things:

1. A way to compute $V(\mathbf{Q})$ and its gradient $\nabla V$.
2. A way to discretize continuous time into finite steps $\Delta t$.

**For the second requirement, we use numerical integration.**

### The Velocity Verlet Algorithm

To solve these continuous equations on a computer, we discretize time using the Velocity Verlet integrator:

$$
\begin{aligned}
    \mathbf{p}_i &\leftarrow \mathbf{p}_i - \frac{1}{2} \nabla_{\mathbf{q}_i} V \cdot \Delta t \\
    \mathbf{q}_i &\leftarrow \mathbf{q}_i + \frac{\mathbf{p}_i}{m_i} \Delta t \\
    \mathbf{p}_i &\leftarrow \mathbf{p}_i - \frac{1}{2} \nabla_{\mathbf{q}_i} V \cdot \Delta t
\end{aligned}
$$

This algorithm is **symplectic**, meaning it approximately conserves the Hamiltonian even with finite $\Delta t$. But there's a catch: stability requires $\Delta t \sim 0.5\text{–}1$ femtoseconds<d-cite key="leach2001timestep"></d-cite>. Larger steps cause energy drift and numerical explosions. **This is the femtosecond prison**—the fundamental timestep barrier that limits all classical MD.

Every loop iteration requires:
- **Computing forces** via $\nabla V$ (expensive)
- **Taking a tiny timestep** (limiting)

We'll address these bottlenecks in sequence, starting with force computation.


<!-- Before navigating the landscape of MLIPs, we must ground ourselves in the Hamiltonian formalism. You can think of the Hamiltonian as the non-negotiable "rulebook" for the system. It is a scalar function that, at any instance, quantifies the total energy of the system. This energy is the sum of two distinct components:

1. **Kinetic Energy ($K$):** The energy of motion, dependent on particle momenta $\mathbf{P}$.
2. **Potential Energy ($V$):** The energy "stored" in atomic configurations and interactions, dependent on particle positions $\mathbf{Q}$.


For an isolated atomistic system with $N$ atoms, masses $m_i$, positions $$\mathbf{Q} = \{\mathbf{q}_i\}_{i=1}^N$$, and momenta $$\mathbf{P} = \{\mathbf{p}_i\}_{i=1}^N$$, the Hamiltonian $H$ takes the following canonical form:

$$
H(\mathbf{P}, \mathbf{Q}) = \underbrace{\sum_{i=1}^N \frac{|\mathbf{p}_i|^2}{2m_i}}_{\text{Kinetic Energy } K(\mathbf{P})} + \underbrace{V(\mathbf{Q})}_{\text{Potential Energy}}
$$

A fundamental law of physics is that in an isolated system (microcanonical ensemble), this Hamiltonian $H$ is conserved ($\frac{dH}{dt} = 0$). This conservation is the constraint our simulation must respect.

How do the atoms "know" how to move to obey this rule? They follow Hamilton's equations of motion:

$$
\frac{d\mathbf{q}_i}{dt} = \frac{\partial H}{\partial \mathbf{p}_i} = \frac{\mathbf{p}_i}{m_i} \quad , \quad \frac{d\mathbf{p}_i}{dt} = -\frac{\partial H}{\partial \mathbf{q}_i}
$$

The first equation simply relates velocity to momentum. The second equation is the physical engine: it states that the time evolution of momentum (the force) is driven strictly by the negative gradient of the potential energy surface (PES), denoted as $V(\mathbf{Q})$. XXX where is V(Q) in the formula?

To solve these continuous equations on a discrete computer, we must turn the infinitely smooth $dt$ into a concrete time step, $\Delta t$. The standard algorithm for this task is the Velocity Verlet (VV) integrator. A single step updates the system as follows:

$$
\begin{aligned}
    \mathbf{p}_i &\leftarrow \mathbf{p}_i - \frac{1}{2} \frac{\partial V}{\partial \mathbf{q}_i} \Delta t \\
    \mathbf{q}_i &\leftarrow \mathbf{q}_i + \frac{\mathbf{p}_i}{m_i} \Delta t \\
    \mathbf{p}_i &\leftarrow \mathbf{p}_i - \frac{1}{2} \frac{\partial V}{\partial \mathbf{q}_i} \Delta t
\end{aligned}
$$

This loop is the beating heart of molecular dynamics. However, to keep this heart beating, we must repeatedly evaluate the gradient term: $\nabla_i V(\mathbf{Q})$. XXX highlight quickly teh force calculation and small times tep prison -->

## Why *ab Initio* Molecular Dynamics Is Expensive

The first bottleneck is evaluating $V(\mathbf{Q})$ and its gradient. Historically, this forced a painful compromise:

- ***Ab initio* methods** (e.g., Density Functional Theory) solve quantum mechanics to compute forces with chemical accuracy. But they require solving the electronic structure problem at every step, with computational cost scaling as $O(N^3)$ or worse—and large constant factors that make even small systems expensive<d-cite key="zhang2018deep"></d-cite>.

- **Classical force fields** use handcrafted functions (harmonic springs, Lennard-Jones potentials, Coulomb terms) that scale linearly with system size. But these predefined functional forms—fixed at design time—cannot adapt to bond breaking, chemical reactions, or complex polarization effects that require quantum mechanical treatment (XXX Source).

For decades, this accuracy-efficiency trade-off defined the field. Quantum accuracy meant tiny systems; large-scale simulations meant sacrificing chemistry.

Machine Learning Interatomic Potentials (MLIPs) changed this.

<!-- The central challenge of MD has always been evaluating this potential $V$. Historically, this forced a painful compromise. On one side, ab initio methods like Density Functional Theory (DFT) offer quantum-mechanical accuracy but scale poorly, restricting simulations to tiny systems and picosecond timescales <d-cite key="zhang2018deep"></d-cite>. 

On the other, classical force fields offer linear scaling ($O(N)$) and speed, but rely on rigid, heuristic approximations that often fail to capture complex chemical reactivity and bond breaking. XXX I dont like this sentence

This accuracy-efficiency trade-off defined the field for decades, until Machine Learning Interatomic Potentials (MLIPs) provided a way to bridge the gap. -->

---
# Machine-Learned Interatomic Potentials: Solving the Force Bottleneck

We've identified two computational bottlenecks in MD. Let's tackle the first one: **computing forces**.

Recall that every timestep requires evaluating $\mathbf{F}_i = -\nabla_i V(\mathbf{Q})$—the gradient of the potential energy surface. Historically, this meant choosing between quantum accuracy (expensive) or classical speed (inaccurate). MLIPs broke this trade-off.

## The MLIP Breakthrough

The core idea is simple: **replace quantum calculations with a learned function**. 

An MLIP is a neural network that maps atomic positions $\mathbf{Q}$ and atomic numbers $\mathbf{Z}$ directly to potential energy:

$$V_\theta(\mathbf{Q}) \approx V_{\text{QM}}(\mathbf{Q})$$

Forces are then obtained via automatic differentiation:

$$\mathbf{F}_{\text{pred}} = -\nabla_{\mathbf{Q}} V_\theta(\mathbf{Q})$$

This bypasses the $O(N^3)$ cost of solving the electronic structure problem at every step.

### Architecture: Graph Neural Networks

Modern MLIPs—such as SchNet<d-cite key="schutt2017schnet"></d-cite>, NequIP<d-cite key="batzner2022nequip"></d-cite>, and MACE<d-cite key="batatia2022mace"></d-cite>—use **Graph Neural Networks** (GNNs) that:

1. **Encode molecular structure naturally**: Atoms are nodes, interactions are edges
2. **Respect physical symmetries**: Predictions are invariant to translation, rotation, and atom permutation
3. **Learn many-body interactions**: Message-passing layers aggregate information from neighboring atoms

The training objective fits both energies and forces jointly:

$$\mathcal{L}(\theta) = \lambda_E \|V_\theta - V_{\text{DFT}}\|^2 + \lambda_F \|\mathbf{F}_{\text{pred}} - \mathbf{F}_{\text{DFT}}\|^2$$

Training on forces directly improves generalization: force supervision provides richer gradient information and helps the model handle out-of-distribution configurations<d-cite key="chmiela2018sgdml"></d-cite>.

### Impact: Quantum Accuracy at Classical Speed

The speedup is dramatic. Where DFT takes **minutes** per force evaluation, MLIP inference takes **milliseconds**—a **1000× improvement**<d-cite key="he2025mlipsbio"></d-cite>. This has enabled:

- **Larger systems**: Million-atom simulations that were previously impossible
- **Longer timescales**: Microsecond trajectories with quantum accuracy
- **New applications**: Drug discovery, battery materials, catalysis<d-cite key="unke2021spookynet"></d-cite>

MLIPs have fundamentally changed what's computationally feasible in molecular simulation.

## The Remaining Challenge: The Femtosecond Prison

MLIPs solved the force bottleneck—**Bottleneck #1**. But **Bottleneck #2** remains stubbornly unsolved.

As we established earlier, classical integrators require $\Delta t \sim 10^{-15}$ s to maintain numerical stability. This means simulating a microsecond—the timescale of protein folding or molecular recognition—still requires $10^9$ sequential steps, regardless of how fast we can compute forces.

**Even with instantaneous force predictions, the serial nature of integration makes long-timescale phenomena computationally intractable.**

To escape the femtosecond prison, we cannot simply accelerate the integrator. We must **bypass it entirely**—replacing step-by-step integration with direct trajectory prediction.

This is where FlashMD enters.
<!-- # Machine-Learned Interatomic Potentials: Solving the Force Bottleneck
We have established that we need billions of time steps to simulate meaningful biological or material phenomena. This brings us to the second half of the computational burden: the cost of a single step.

As derived in the Hamiltonian framework, every single step requires us to evaluate the gradient of the potential energy surface: $\mathbf{F}_i = -\nabla_i V(\mathbf{Q})$.

## The MLIP Breakthrough

At its core, an MLIP is a regression framework that approximates the Potential Energy Surface (PES) by learning a mapping $\mathcal{F}_\theta: (\mathbf{Z}, \mathbf{Q}) \to \mathbb{R}$ from atomic numbers and coordinates directly to the scalar potential energy. This effectively bypasses the $O(N^3)$ cost of solving the electronic structure explicitly.

Unlike classical force fields, MLIPs leverage deep neural networks—typically Graph Neural Networks (GNNs) or Message Passing Neural Networks (MPNNs)—to serve as universal approximators of the quantum mechanical interaction <d-cite key="he2025mlipsbio"></d-cite>. This allows them to capture complex, non-local many-body effects that classical approximations inherently miss.

Crucially, MLIPs enforce physical consistency by defining atomic forces as the exact negative gradient of the predicted energy with respect to atomic positions via automatic differentiation:

$$\mathbf{F}_{\text{pred}} = -\nabla_{\mathbf{Q}} E_{\text{pred}}(\mathbf{Q})$$

The training objective is therefore a multi-task learning problem. We optimize the network parameters $\theta$ to minimize a composite loss against ground-truth quantum mechanical labels (typically from DFT):

$$\mathcal{L}(\theta) = \lambda_E \|E_{\text{pred}} - E_{\text{DFT}}\|^2 + \lambda_F \|\underbrace{-\nabla_{\mathbf{Q}} E_{\text{pred}}}_{\mathbf{F}_{\text{pred}}} - \mathbf{F}_{\text{DFT}}\|^2$$

By training on high-quality snapshots of molecular configurations—generated via random sampling or active learning—MLIPs can capture complex, many-body interactions that classical methods simply cannot see. XXX source

## Why Faster Forces Are Still Not Enough
Machine Learning Interatomic Potentials (MLIPs) have revolutionized the field by reducing the computational cost of force calculation ($F$) by orders of magnitude compared to DFT. However, they leave the fundamental architectural flaw of MD untouched: the integrator bottleneck.

Classical integrators like Velocity Verlet face a hard physical speed limit. To maintain numerical stability and energy conservation, the time step $\Delta t$ must resolve the fastest atomic vibrations in the system—typically the oscillation of hydrogen bonds. This confines simulations to the femtosecond scale ($\Delta t \approx 10^{-15} \text{s}$), regardless of how fast the force model is.

This creates a massive discrepancy between simulation time and biological reality. To simulate a mere microsecond of physical time—relevant for protein folding or drug binding—we must perform one billion sequential steps:

$$N_{\text{steps}} = \frac{10^{-6} \text{ s}}{10^{-15} \text{ s}} = 10^9 \text{ steps}$$

We are effectively trapped in a "femtosecond prison." Even with instant force predictions, this serial dependency makes long-timescale phenomena computationally intractable. To escape this, we cannot simply accelerate the integrator; we must bypass it entirely. -->

---


# FlashMD: Escaping the Femtosecond Prison

FlashMD introduces a transformative approach: rather than incrementally integrating forces like a standard force field, it operates as a direct trajectory predictor.

Instead of acting as a "middleman"—predicting energy to derive forces for a classical integrator—FlashMD learns the dynamical map directly from simulation data:

$$\mathcal{G}_\theta: (\mathbf{Q}_t, \mathbf{P}_t) \to (\mathbf{Q}_{t+\Delta t}, \mathbf{P}_{t+\Delta t})$$

This paradigm shift enables the model to predict the system’s next state in a single forward pass, replacing hundreds of small integration steps and allowing for strides 1-2 magnitudes larger than the stability limit of numerical integrators.

XXX Here Illustration of Classical MD Loop vs. FlashMD Loop

## Architecture and Design Principles of FlashMD 
<!-- The FlashMD Architecture -->
To realize the dynamical map $\mathcal{G}_\theta$ defined above, FlashMD implements a flexible deep learning pipeline. While the current implementation defaults to a specific Transformer backbone, the architecture is fundamentally modular.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/flashmd_architecture.jpeg" class="img-fluid" %}

We can view FlashMD as a wrapper that prepares atomic data for any powerful graph neural network:

1. **Input: Embedding the Atomic State** The raw atomic state is converted into a graph representation.
   - The current positions $\mathbf{Q}$ and momenta $\mathbf{P}$ are encoded into node and edge features of a molecular graph.
   - **Mass Scaling:** Input momenta are normalized ($\tilde{\mathbf{p}}_i = \mathbf{p}_i / \sqrt{m_i}$) to prevent heavy atoms from dominating the loss, ensuring the model captures fast hydrogen vibrations as accurately as heavy-atom motions.

2. **Backbone: Point-Edge Transformer (PET)** The graph is processed by a message-passing network to extract local geometric features. FlashMD uses the Point-Edge Transformer (PET) by default, which updates edge and node representations via attention mechanisms. At inference time, optional filters can be applied: momentum rescaling for energy conservation, thermostat/barostat integration for ensemble control, and random rotations to mitigate symmetry-breaking artifacts.

3. **Output: Multi-Head Prediction** Two separate MLP heads branch from the final node representations to predict the update:
     
   - Momentum Head: Predicts $\mathbf{p}_i(t + \Delta t)$.
   - Displacement Head: Predicts $\Delta \mathbf{q}_i(t + \Delta t)$.
  
Theoretically, you could swap the backbone for any modern GNN. However, molecular dynamics imposes a strict "non-negotiable" constraint that narrows our choices significantly: E(3) Equivariance.

## The Symmetry Challenge: E(3) Equivariance
Imagine simulating a water molecule. If you rotate your entire simulation box by 90 degrees, the physics must remain identical. The potential energy should not change, and the force vectors must rotate by exactly 90 degrees to match the atoms.

Standard neural networks see coordinates as simple lists of numbers; they do not inherently "know" that a rotated molecule is the same physical object. If we denote our model as $\mathcal{F}$ and a rotation matrix as $\mathcal{R}$, the model must satisfy:

$$
\mathcal{F}(\mathcal{R} \cdot \mathbf{Q}) = \mathcal{R} \cdot \mathcal{F}(\mathbf{Q})
$$

If a model fails this test, it might predict that a molecule flies apart simply because it was rotated to face "North" instead of "East."

There are generally two ways to solve this in Deep Learning:

1. Hard Constraints (e.g., NequIP<d-cite key="NequIP_Batzner2022"></d-cite>, MACE<d-cite key="MACE_ALLEGRO_leimeroth2025machine"></d-cite>): Bake geometric algebra (spherical harmonics) directly into the network layers. This guarantees exact equivariance but is computationally expensive.

2. Soft Constraints (e.g., SchNet<d-cite key="schutt2017schnet"></d-cite>, PET<d-cite key="PET_pozdnyakov2023smooth"></d-cite>): Use a flexible, standard architecture and "teach" it symmetry through data augmentation or frame averaging.


## The Point-Edge Transformer (PET)

- is a rotationally unconstrained and transformer-based graph neural network
- PET maintains feature vectors (or messages) f_l ij for every directed bond between atoms i and j that lie within a specified cutoff radius.
- These intermediate representations are updated at each message-passing layer by a transformer
- outputs are subsequently interpreted as the new set of outbound messages from atom i to each neighbor j
- geometric information and chemical species are also incorporated
- A feed-forward NN is used to obtain the desired output/target property
- PET architecture imposes no explicit rotational symmetry constraints, but learns to be equivariant through data augmentation.
- This unconstrained approach yields high theoretical expressivity: even a single layer of the model acts as a universal approximator featuring virtually unlimited body order and angular resolution


The PET architecture<d-cite key="PET_pozdnyakov2023smooth"></d-cite> reimagines atomic interactions through the lens of modern Transformers. While standard message-passing GNNs aggregate neighbor information via simple summation, PET introduces a richer mechanism that naturally captures **many-body correlations**—the complex ways in which multiple neighbors jointly influence a central atom.

(XXX add illustration of PET)

The key innovations are:

1. **Tokenization of neighbors.** For each central atom, every neighbor within a cutoff radius $R_c$ is encoded into a distinct *abstract token* that carries both geometric (relative position) and chemical (species) information. Unlike standard GNNs that collapse neighbor information into a single aggregated vector, PET preserves the identity of each interaction.

2. **Self-attention over interactions.** These tokens are processed by a Transformer-style self-attention mechanism. This allows the model to learn that the presence of one neighbor dynamically modifies the effective interaction with another—for example, how a third oxygen atom weakens a hydrogen bond between two water molecules. These **many-body effects** emerge naturally from attention, without requiring hand-crafted descriptors.

3. **Computational efficiency.** By avoiding the expensive mathematical machinery of spherical harmonics and Clebsch–Gordan coefficients required by strictly equivariant architectures, PET achieves competitive accuracy at significantly lower computational cost. The price is that rotational symmetry must be learned rather than guaranteed.

### Enforcing Symmetry at Runtime


XXX check section and shorten it
Since PET is not intrinsically equivariant, FlashMD must enforce symmetry through two complementary mechanisms.

**During training**, random rotations are applied to every training sample (**data augmentation**). This teaches the model to produce consistent predictions regardless of molecular orientation—but the resulting equivariance is only approximate, limited by finite training data and model capacity.

**During inference**, FlashMD adds a second layer of protection: **Stochastic Frame Averaging**. Before each prediction step:

1. The entire system $(\mathbf{Q}, \mathbf{P})$ is rotated by a random matrix $\mathcal{R}$.
2. The backbone computes the next state in the rotated frame.
3. The output is mapped back to the original frame via $\mathcal{R}^{-1}$.

$$\mathbf{y}_{\text{final}} = \mathcal{R}^{-1} \cdot \mathcal{F}_{\theta}(\mathcal{R} \cdot \mathbf{x})$$

Over many rollout steps, this ensures that predictions are **statistically invariant** to orientation—any systematic bias toward a particular direction averages out. It is a pragmatic compromise: cheaper than the full Equivariant Coordinate System Ensemble (ECSE) proposed in the original PET paper, but sufficient for the long rollouts FlashMD targets.

The combination of data augmentation and stochastic frame averaging makes PET *practically* equivariant—not by mathematical proof, but by empirical convergence. Whether this approximation is good enough depends on the application. For the thermostatted benchmarks in the next section, it works remarkably well. For the stricter NVE setting in our [exploratory study](#an-exploratory-study-on-failure-modes), even small symmetry violations may compound.




## Long-Stride Predictions in Practice
To demonstrate what FlashMD is capable of, the authors evaluate it across a diverse set of benchmarks and experiments, designed to answer a simple question:
Can we simulate realistic molecular dynamics with much larger time steps – without losing essential physics?

FlashMD is explored in two flavors:

- A water-specific model, trained only on liquid water
- A universal model, trained on chemically diverse systems, meant to generalize across molecules and materials

This lets the authors study both ends of the spectrum: maximum accuracy for one system, and broad applicability across many.


### The Testing Strategy
Before diving into results, it's worth understanding how you even benchmark a method like FlashMD. A direct, step-by-step comparison of trajectories is impossible: MD is chaotic – tiny differences grow exponentially, so two simulations will quickly diverge even if both are “correct”.

Instead, the researchers took a statistical approach:

1. Generate reference trajectories using conventional MD with a reliable force field (PET-MAD)
2. Run FlashMD simulations under the same conditions
3. Compare statistical properties rather than individual trajectories—things like density, radial distribution functions (how atoms arrange themselves around each other), and phase transition temperatures

As the authors note: "we primarily focus our quantitative analysis on time-independent equilibrium properties, and discuss examples where FlashMD qualitatively captures time-dependent behavior." In other words: check if the average properties match, and see if the dynamics look qualitatively reasonable.

### Key Results at a Glance

**1. Liquid Water:**
Water might seem simple, but it's notoriously tricky to simulate due to its hydrogen bonding network. The team tested both their water-specific and universal models on liquid water at 450 K (above the melting point for their particular model).

**Key findings:**
- Temperature control worked well when using appropriate thermostats (Langevin), with deviations typically under 1 K from target temperature
- Radial distribution functions (which show how oxygen and hydrogen atoms arrange themselves) matched reference MD simulations nearly perfectly
- Density predictions from constant-pressure simulations were accurate for water-specific models and reasonable for universal models
- Models could handle strides up to 16 fs—a 64× speedup compared to the 0.25 fs timesteps typically needed

**2. Solvated Alanine Dipeptide: Protein-like Dynamics**
This system—a small peptide in water—serves as a minimal model for protein flexibility. The critical test: can FlashMD capture the Ramachandran plot, which maps out the backbone conformations proteins can adopt?

Remarkably, this works even with strides up to 32× larger than standard MD time steps – a strong indication that FlashMD preserves meaningful molecular motion, not just static snapshots.

**3. Aluminum Surface: Catching Pre-melting Phenomena**
Metal surfaces exhibit fascinating behavior at high temperatures: atoms start becoming mobile before bulk melting occurs, a phenomenon called pre-melting. This requires capturing subtle, layer-specific dynamics.

**Key findings:**
- Correctly reproduced the anisotropic softening pattern—surface atoms wiggling more in one direction, second-layer atoms in another
- Captured dynamic defect formation: temporary creation and migration of surface atoms
- Achieved this with 64 fs strides (64× faster than the 1 fs baseline), while still showing physically meaningful atomic trajectories

This shows that FlashMD is not limited to molecules, but can handle complex solid-state phenomena.

**4. Lithium Thiophosphate: Superionic Phase Transitions**
Perhaps the most impressive demonstration involved a solid-state battery electrolyte material. At high temperatures, lithium atoms become highly mobile in a "superionic" state—critical for battery performance.

The challenge: Predict temperature-dependent lithium conductivity and capture the phase transition.

**Key findings:**
- Successfully predicted the superionic transition temperature at 675 K, within the expected range
- Reproduced the dramatic increase in lithium ion conductivity across the transition
- Some systematic errors appeared (over/underestimation at low/high temperatures), but the overall behavior was captured with 8× speedup

Here, FlashMD demonstrates that it can capture slow, collective processes, which are traditionally hard to access with MD.


---

# Limitations and Open Challenges
FlashMD promises incredible speed ($100\times$), but we have to ask: is the physics still real? Neural networks are pattern matchers, not physics engines. They don't actually understand laws like energy conservation; they only memorize the data they’ve seen. When we push the model beyond its training distribution, cracks begin to appear.

## Learning Dynamics Without Physical Guarantees

Velocity Verlet comes with mathematical guarantees: energy conservation, symplecticity, time-reversibility. A learned model has no such guarantees. Let's examine what can go wrong.

### 1. Out-of-Distribution Drift

FlashMD is trained on equilibrium MD trajectories. But during a long rollout, small errors compound. After 1,000 steps (160 ps with 16 fs strides), the system may drift into configurations never seen during training.

Unlike MLIPs—which predict one step ahead—FlashMD's errors accumulate **autoregressively**. This demands robust uncertainty quantification: the model must know when it doesn't know.

### 2. Chaotic Dynamics

Molecular systems are chaotic: nearby trajectories diverge exponentially (Lyapunov exponent). This imposes a fundamental limit—even a perfect model cannot predict beyond the system's decorrelation time (typically picoseconds for liquids, nanoseconds for proteins).

This isn't a bug; it's physics. FlashMD must capture the **statistical ensemble** of trajectories, not a single deterministic path. This introduces **aleatoric uncertainty**—irreducible randomness from the chaotic dynamics itself.

### 3. Energy Conservation

Here's the Achilles' heel. Velocity Verlet conserves energy to machine precision. A neural network has no such constraint—small prediction errors cause energy drift:

$$\Delta H = H(\mathbf{Q}_{t+\Delta t}, \mathbf{P}_{t+\Delta t}) - H(\mathbf{Q}_t, \mathbf{P}_t)$$

FlashMD addresses this two ways:

1. **Training:** Include $\|\Delta H\|$ in the loss function, encouraging implicit energy conservation
2. **Inference:** Rescale momenta post-prediction to enforce $H_{t+\Delta t} = H_t$ exactly

The second approach is aggressive but necessary. However, it modifies the dynamics—we're no longer solving Hamilton's equations, but a constrained variant. Does this preserve the correct statistical ensemble? (We'll return to this question.)

### 4. Symplectic Structure

In Hamiltonian mechanics, phase space has geometric structure (symplecticity) that preserves volume. Classical integrators respect this; neural networks don't.

Enforcing symplecticity explicitly—via a generating function parameterization—is theoretically possible but computationally prohibitive (it requires computing a 3N × 3N Jacobian). FlashMD takes a pragmatic approach: train on symplectic data (VV trajectories) and hope the model learns the structure implicitly.

This is a gamble. Without explicit enforcement, thermodynamic properties (temperature, pressure, free energies) may drift over long timescales.

### 5. Rotational Symmetry

Physics is rotationally invariant: if you spin your simulation box, the dynamics shouldn't change. A standard neural network doesn't "know" this—it must learn it from data.

FlashMD shows it can maintain physical accuracy while taking dramatically larger steps through time, across a diverse range of materials. But as we'll see next, this performance comes with important caveats—particularly around energy conservation...

FlashMD mitigates this via:
- **Data augmentation:** Random rotations during training
- **Runtime augmentation:** Random rotations at each prediction step


# An Exploratory Study on Failure Modes

To move beyond theoretical concerns, we conduct a systematic exploratory study on a concrete system: a periodic box of 258 TIP3P water molecules simulated with OpenMM as ground truth. We train FlashMD models under varying conditions and evaluate their ability to conserve energy during NVE rollouts—the most unforgiving test of physical validity.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/tip3p_maxwell_boltzmann_comparison.png" class="img-fluid" label="fig:maxwell" caption="Figure 3: Comparison of Maxwell-Boltzmann distributions for different temperatures" %}

## Experimental Setup

**Ground Truth Generation.** We generate NVE trajectories at different temperatures (200K–700K, 20K steps) using the TIP3P force field in OpenMM, saving configurations every 0.5 fs for 10 ps each. We verify thermodynamic consistency by comparing the velocity distributions against the Maxwell-Boltzmann distribution at each temperature ([Figure 3](#fig:maxwell)).

**Training Data.** From these trajectories, we construct FlashMD training pairs $(q_t, p_t) \to (q_{t+\Delta t}, p_{t+\Delta t})$ at a prediction stride of $\Delta t = 1\,\text{fs}$. Following the original paper, targets are stored as mass-scaled quantities: $\Delta\tilde{q}_i = \Delta q_i \sqrt{m_i}$ and $\tilde{p}_i = p_i / \sqrt{m_i}$. XXX check again this in code

**Evaluation Protocol.** Each trained model is deployed in NVE simulation for 50 ps (50,000 steps at 1 fs) starting from a 300 K equilibrated configuration. We track total energy $E_\text{tot}(t)$, temperature $T(t)$, and energy drift rate $\dot{E}$ (eV/ps). A model is considered "exploded" if $T > 1000\,\text{K}$ at any point.

XXX before ablation show the trained models are stable but lack the energy conservation leading to huge energy and temperature drifts.

The first results of the simulations depict in Figure XX. We trained the models accordingly to the setup above. We also investigated different starting points from the NVE groundtruth to see the bahaviour od the model as well as different seeds to handle stochastic training varaicen. All custom models show that all of the trained models achieve stable NVE simulations. But all of them lack the energy conservation drastically. We see huge drifts in energy and temperature. This indicates us that the model loses some energy somehow. To understand this behaviour we also want to take a look at the momenta distribution of the models compared to the ground truth. Since the momenta relates directly to the potential energy this is traight forward.

XXX Image of plots fro the distributions here.

As we can see. All of custom models exhibit extensive differences in the momenta of Oxygen atoms. But interestingly we can see that if turn on the rescaling filter we can even better distributions. This leads us to question how could we improve the momentum distrubtion of the model by doing two things:
- Loss weighting between positons and momenta
- Applying mass sacling to the positons and momenta

## Ablation 1: Loss Weighting Between Positions and Momenta

The first question is simple: does it matter how much the model cares about getting momenta right vs. positions? We train models with momentum loss weights $w_p \in \{0.5, 1.0, 1.5, 2.0, 10.0\}$ while keeping the position weight fixed at $w_q = 1.0$, and let them run NVE for 10 ps.

The results are... surprising.

| Model | $w_p$ | Drift (eV/ps) | $\bar{T}$ (K) | $\sigma_T$ (K) | Stable until |
|-------|-------|----------------|---------------|-----------------|--------------|
| $w_p=0.5$ | 0.5 | 2813.95 | 513.8 | 148.9 | 8.1 ps ✗ |
| $w_p=1.0$ (Baseline) | 1.0 | -3.70 | 227.1 | 34.2 | >10 ps ✓ |
| $w_p=1.5$ | 1.5 | -4.80 | 178.0 | 38.8 | >10 ps ✓ |
| $w_p=2.0$ | 2.0 | 52.80 | 476.1 | 168.5 | 2.7 ps ✗ |
| $w_p=10.0$ | 10.0 | 4.39 | 334.9 | 67.5 | >10 ps ✓ |
| Ground Truth | — | -0.0035 | 304.7 | 6.9 | >10 ps ✓ |

{% include figure.liquid path="assets/img/2026-04-27-flashmd/nve_comparison_4models_10ps_temp_energy.png" class="img-fluid" label="fig:ablation1" caption="Figure X: NVE trajectories for three momentum loss weights and OpenMM ground truth. Top: temperature evolution. Bottom: total energy. The ground truth (black) is essentially flat — none of the models come close." %}

If you expected "more momentum weight = better momenta = more stable," you'd be wrong. The relationship is wildly non-monotonic, and we can group the results into three regimes:

**Too little ($w_p = 0.5$): immediate explosion.** The model doesn't learn momentum dynamics well enough and blows up within 8 ps. Temperature rockets past 500 K, energy drift is off the charts. Lesson: you *need* to care about momenta.

**The "moderate" zone ($w_p = 1.0, 1.5$): stable but freezing.** These models survive the full 10 ps — but look at the temperatures! The baseline cools from 300 K to a mean of 227 K, and $w_p = 1.5$ cools even further to 178 K. The model is systematically predicting momenta that are too small. It's not exploding, but it's not doing physics either — it's slowly bleeding kinetic energy. You can see this clearly in [Figure X](#fig:ablation1): the blue and green lines drift steadily downward while the ground truth holds at ~305 K.

**Cranked to 10 ($w_p = 10.0$): closest to reality, but noisy.** Here's the twist: the most extreme weight actually produces the most realistic temperature (335 K, just 10% above target). Looking at [Figure X](#fig:ablation1), the red $w_p=10$ line tracks the ground truth best during the first ~5 ps. But the fluctuations are huge ($\sigma_T = 67$ K vs. ground truth's 7 K), and by 8 ps it starts slowly heating up.

To our suprise, $w_p = 2.0$ explodes at 2.7 ps. Somehow, 2× the baseline is catastrophically unstable while 10× is fine. The loss landscape clearly has some sharp cliffs in this region.

**The bottom line:** Look at the energy panel in [Figure X](#fig:ablation1). The ground truth (black line) is essentially a flat line. Every single model drifts visibly — the best ones still have energy drift **1000× larger** than OpenMM. Loss weighting can shift the failure mode from cooling to heating, but it can't bridge that gap.

This tells us something important: the problem isn't *how much* the model cares about momenta — it's *how* it represents them. Which brings us to our next experiment: what if we change the loss to account for the fact that hydrogen and oxygen have very different masses?

much* the model cares about momenta, we change *how* it represents them through mass-scaled loss formulations. -->


## Ablation 2: Mass-Scaled Loss Functions
Standard MSE treats all atoms equally in momentum space. But since $p_i = m_i v_i$, oxygen atoms (mass 16) dominate the momentum loss by a factor of $16^2 = 256$ compared to hydrogen (mass 1). This means the model optimizes primarily for oxygen momenta while hydrogen velocity errors remain large.

Following the FlashMD paper, we implement a mass-scaled loss:

$$\mathcal{L}_{\tilde{p}} = \frac{1}{N} \sum_i \frac{\|p_i^\text{pred} - p_i^\text{true}\|^2}{m_i}, \qquad \mathcal{L}_{\Delta\tilde{q}} = \frac{1}{N} \sum_i \|\Delta q_i^\text{pred} - \Delta q_i^\text{true}\|^2 \cdot m_i$$

This is equivalent to computing MSE on the mass-scaled quantities $\tilde{p} = p/\sqrt{m}$ and $\Delta\tilde{q} = \Delta q \cdot \sqrt{m}$, ensuring that velocity errors are weighted equally regardless of atomic mass.

{% include figure.liquid path="assets/img/2026-04-27-flashmd/mass_scaling_nve_comparison.png" class="img-fluid" label="fig:mass-scaling" caption="Figure X: NVE trajectory comparison between standard MSE and mass-scaled MSE loss. Left column shows the initial drift phase (0–10 ps), right column the full 50 ps trajectory. Both models exhibit systematic cooling relative to the ground truth." %}

| Model | Loss Type | Drift (eV/ps) | $\bar{T}$ (K) | $\sigma_T$ (K) | $\sigma_E$ (kJ/mol) | Stable until |
|-------|-----------|----------------|---------------|-----------------|---------------------|--------------|
| Custom Trained | MSE | -0.6842 | 207.1 | 23.0 | 591.1 | >50 ps |
| Mass-Scaled Custom Trained | mass\_scaled\_mse | -0.6989 | 212.9 | 24.9 | 642.0 | >50 ps |
| Ground Truth (OpenMM) | — | -0.0003 | 304.2 | 6.9 | 1.2 | >50 ps |

**Key finding:** Mass-scaling provides a modest improvement in the early drift phase but does not fundamentally alter long-term stability. A time-windowed analysis reveals the nuance:

- **0–5 ps:** Mass-scaling reduces energy drift by 26% (-3.72 vs. -5.03 eV/ps) and keeps the temperature closer to 300 K (272 K vs. 254 K, $\sigma_T$ halved from 26 to 13 K). The model clearly benefits from balanced treatment of H and O atoms in the initial phase.
- **0–10 ps:** The advantage persists but narrows—drift is reduced by 24% (-2.82 vs. -3.70 eV/ps), mean temperature 252 K vs. 227 K.
- **0–50 ps:** Both models converge to nearly identical behavior—drift rates of -0.68 and -0.70 eV/ps, mean temperatures of 207 K and 213 K. The early advantage is fully washed out.

**Physical interpretation:** Mass-scaling correctly addresses the *symptom*—unequal error weighting across species—but not the *disease*. Both models systematically cool from 300 K to ~210 K over 50 ps, losing roughly $\frac{1}{3}$ of their kinetic energy. This cooling pattern ($\dot{E} \approx -0.7$ eV/ps, $\sigma_E \sim 600$ kJ/mol vs. ground truth $\sigma_E = 1.2$ kJ/mol) represents a $500\times$ violation of energy conservation that no loss reweighting can fix.

The fact that mass-scaling helps *early* but not *late* suggests that it improves the model's initial momentum predictions (reducing the per-species bias), but the accumulated errors from the autoregressive rollout eventually dominate regardless. The fundamental issue is that FlashMD's single-step prediction errors, while small individually, compound systematically rather than canceling stochastically—a hallmark of non-symplectic integration.

## Summary of Findings

Our exploratory study reveals a consistent pattern across all model variants:

1. **All models exhibit systematic energy drift**, even in the quasi-stable 
   regime. This is expected—the model has no symplectic structure or 
   energy-conserving inductive bias.

2. **Loss weighting affects the drift rate** but does not eliminate it. 
   Mass-scaled losses with appropriate momentum weighting achieve the 
   best short-term stability.

3. **Failure is always catastrophic once it begins.** There is no graceful 
   degradation—once the system leaves the training distribution, errors 
   compound exponentially.

These findings suggest that improving the loss function alone is insufficient. The fundamental challenge is distributional: the model must either (a) be trained on data hat covers the states it will visit during long rollouts, or (b) incorporate physical constraints that prevent drift toward unphysical regions.

---

# Conclusion

In this post, we traced the two fundamental bottlenecks of molecular dynamics and showed how each has been addressed: MLIPs solved the force bottleneck; FlashMD bypasses the integrator entirely, predicting system evolution at strides one to two orders of magnitude beyond classical stability limits. Across water, proteins, metals, and battery electrolytes, it recovers the correct statistical physics at speedups of 8× to 64×.

But our independent exploratory study reveals a sobering counterpoint. When we strip away the thermostats that regulate temperature in standard benchmarks, the learned dynamics fail to conserve energy. Every model we tested—regardless of loss weighting or mass scaling—exhibits systematic energy drift orders of magnitude larger than classical integrators. The problem is structural, not parametric: no amount of loss function tuning resolved it.

This does not diminish FlashMD's contribution—it is the first model to learn meaningful long-stride dynamics across chemically diverse systems. But it makes clear what is still missing: the mathematical guarantees that classical integrators provide for free. Until symplectic architectures, hybrid corrective schemes, or active learning strategies close this gap, FlashMD is best understood as a powerful tool for thermostatted simulations rather than a general-purpose replacement for classical integrators.

The question we posed in the introduction—*Can a model learn to respect the laws of physics without being explicitly taught to do so?*—now has a nuanced answer. For statistical properties sampled under external control, yes. For the strict, unassisted conservation of energy that defines Hamiltonian dynamics, not yet.



XXX Propose a way, scetch a solution maybe add first experiments with handselected active learning ...
---

# Conclusion
*(Your text here...)*
