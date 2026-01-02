# **Chapter 5: Quantum Fourier Transform and Phase Estimation**

---

The goal of this chapter is to establish concepts in the Quantum Fourier Transform (QFT) and Quantum Phase Estimation (QPE). The QFT is a key component in many quantum algorithms, and QPE uses it to determine the eigenvalues of unitary operators.

---


## **5.1 Quantum Fourier Transform (QFT)** {.heading-with-pill}

> **Difficulty:** ★★☆☆☆
> 
> **Concept:** Basis change to the frequency domain via phase gradients
> 
> **Summary:** The QFT maps computational-basis amplitudes to a Fourier basis. For $N=2^n$ it acts as $|j\rangle\mapsto\tfrac{1}{\sqrt{N}}\sum_{k=0}^{N-1}\omega_N^{jk}|k\rangle$ with $\omega_N=e^{2\pi i/N}$ and admits a circuit of $\mathcal{O}(n^2)$ one- and two-qubit gates.

### **Theoretical Background**

The Quantum Fourier Transform (QFT) is the quantum analog of the discrete Fourier transform (DFT), mapping computational basis states to a superposition with phase relationships encoding frequency information.

**Mathematical Definition:**  
For an $n$-qubit system with dimension $N = 2^n$, define the primitive $N$-th root of unity:

$$
\omega_N = e^{2\pi i/N}
$$

The QFT acts on computational basis states as:

$$
\text{QFT}_N|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} \omega_N^{jk}|k\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle
$$

By linearity, for an arbitrary state $|\psi\rangle = \sum_{j=0}^{N-1} \alpha_j|j\rangle$:

$$
\text{QFT}_N|\psi\rangle = \sum_{j=0}^{N-1} \alpha_j \cdot \text{QFT}_N|j\rangle = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1}\sum_{k=0}^{N-1} \alpha_j e^{2\pi ijk/N}|k\rangle = \sum_{k=0}^{N-1} \tilde{\alpha}_k|k\rangle
$$

where $\tilde{\alpha}_k = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} \alpha_j e^{2\pi ijk/N}$ are the Fourier coefficients.

**Product Representation:**  
Expressing basis states in binary $j = j_{n-1}2^{n-1} + \cdots + j_1 2 + j_0$ with $j_\ell \in \{0,1\}$, the QFT has the elegant product form:

$$
\text{QFT}_N|j_{n-1}\cdots j_1 j_0\rangle = \frac{1}{2^{n/2}}\bigotimes_{\ell=0}^{n-1} \left(|0\rangle + e^{2\pi i \cdot 0.j_\ell j_{\ell-1}\cdots j_0}|1\rangle\right)
$$

where $0.j_\ell j_{\ell-1}\cdots j_0$ denotes the binary fraction $\sum_{m=0}^\ell j_m 2^{-(\ell-m+1)}$.

**Circuit Decomposition:**  
This product structure enables an efficient quantum circuit using:

1. **Hadamard gates** on each qubit to create superposition  
2. **Controlled phase rotations** $R_m = \text{diag}(1, e^{2\pi i/2^m})$ between qubits  
3. **Bit-reversal swaps** at the end

The complete circuit for qubit $j$ (indexed from $n-1$ down to $0$):

$$
\text{QFT}_N = \text{SWAP}_{\text{reverse}} \cdot \left(\prod_{j=n-1}^{0} \mathbf{H}_j \prod_{k=j+1}^{n-1} R_{k-j+1}^{(k,j)}\right)
$$

where $R_m^{(k,j)}$ denotes controlled-$R_m$ with control on qubit $k$ and target on qubit $j$.

**Gate Complexity:**  
The circuit requires:
- $n$ Hadamard gates  
- $\sum_{j=0}^{n-1}(n-1-j) = \frac{n(n-1)}{2}$ controlled rotations  
- $\lfloor n/2 \rfloor$ swap gates

Total: $\mathcal{O}(n^2)$ gates, exponentially better than the classical FFT's $\mathcal{O}(N\log N) = \mathcal{O}(n \cdot 2^n)$ operations.

**Inverse QFT:**  
The inverse operation is:

$$
\text{QFT}_N^{-1} = \text{QFT}_N^\dagger
$$

implemented by reversing gate order and conjugating rotations: $R_m \to R_m^\dagger = R_m^{-1}$.

### **Comprehension Check**

!!! note "Quiz"
    **1. What is $\omega_N$ in the QFT definition?**

    - A. A normalization constant  
    - B. A twiddle factor/root of unity  
    - C. A probability amplitude  
    - D. A diffusion parameter  

    ??? info "See Answer"
        **Correct: B.** $\omega_N=e^{2\pi i/N}$.

    **2. Asymptotic gate count of a straightforward QFT circuit on $n$ qubits is:**

    - A. $\mathcal{O}(n)$  
    - B. $\mathcal{O}(n^2)$  
    - C. $\mathcal{O}(2^n)$  
    - D. $\mathcal{O}(n\log n)$  

    ??? info "See Answer"
        **Correct: B.** Using Hadamards and controlled rotations yields $\mathcal{O}(n^2)$ gates.

-----

!!! abstract "Interview-Style Question"

    **Q:** Why does the QFT not, by itself, provide a general exponential speedup when compared to the classical FFT?

    ???+ info "Answer Strategy"
        **The QFT Transformation:**  
        The Quantum Fourier Transform implements $\text{QFT}|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle$ using $\mathcal{O}(n^2)$ gates, exponentially faster than classical FFT's $\mathcal{O}(N \log N)$ operations. However, this apparent speedup has a critical limitation.
        
        $$
        \text{QFT}|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle
        $$

        **The Measurement Bottleneck:**  
        While QFT efficiently transforms amplitudes $|\psi\rangle = \sum_j \alpha_j|j\rangle \to \sum_k \tilde{\alpha}_k|k\rangle$, measurement collapses to a single outcome $|k_0\rangle$ with probability $|\tilde{\alpha}_{k_0}|^2$. Extracting all $N$ Fourier coefficients requires $\mathcal{O}(N)$ measurements, negating the speedup. Classical FFT outputs all coefficients explicitly in $\mathcal{O}(N \log N)$ operations.

        **When QFT Provides Advantage:**  
        QFT becomes powerful when embedded in larger algorithms exploiting interference patterns before measurement. QPE uses QFT to concentrate probability into binary phase representations. Shor's algorithm creates peaks at $k = 0, N/r, 2N/r, \ldots$ enabling period extraction. HHL manipulates eigenvalue-dependent phases.
        
        **Conclusion:**  
        QFT alone doesn't replace classical FFT due to measurement limitations. Exponential advantage emerges when QFT enables algorithmic structures that concentrate probability into polynomially many outcomes, making measurement informative.

-----

### **<i class="fa-solid fa-flask"></i> Hands-On Projects**

#### **Project Blueprint**
| **Section**              | **Description** |
| ------------------------ | --------------- |
| **Objective**            | Compute QFT on $N=4$ for input $\|1\rangle$ and interpret phases. |
| **Mathematical Concept** | DFT over $\mathbb{Z}_4$; roots of unity. |
| **Experiment Setup**     | Two qubits; basis $\\{\|00\rangle,\|01\rangle,\|10\rangle,\|11\rangle\\}$. |
| **Process Steps**        | Apply $\mathrm{QFT}_4\|1\rangle=\tfrac{1}{2}\sum_{k=0}^3 e^{2\pi i k/4}\|k\rangle$; expand coefficients. |
| **Expected Behavior**    | Uniform magnitudes $1/2$; phases progress linearly: $1,i,-1,-i$. |
| **Tracking Variables**   | Complex amplitudes $c_k$; probabilities $|c_k|^2$. |
| **Verification Goal**    | Check normalization and phase pattern. |
| **Output**               | Statevector entries and interpretation. |

#### **Pseudocode Implementation**
```pseudo-code
FUNCTION Compute_QFT(initial_state_vector):
    # Assert input is a valid quantum state for N=4 (2 qubits)
    ASSERT Is_Valid_State(initial_state_vector, num_qubits=2)

    # Define the QFT matrix for N=4
    N = 4
    omega = exp(2 * PI * 1j / N)
    QFT_matrix = [[omega**(j*k) for k in 0..N-1] for j in 0..N-1] / sqrt(N)
    LOG "QFT_4 matrix constructed."

    # Step 1: Apply the QFT matrix to the initial state |1> (vector [0,1,0,0])
    final_state_vector = Matrix_Vector_Multiply(QFT_matrix, initial_state_vector)
    LOG "Final state vector: ", final_state_vector
    # Expected: 0.5 * [1, i, -1, -i]

    # Step 2: Extract amplitudes and phases from the final state vector
    amplitudes = [abs(c) for c in final_state_vector]
    phases = [arg(c) for c in final_state_vector]
    LOG "Amplitudes: ", amplitudes
    LOG "Phases (in radians): ", phases

    # Step 3: Verify the properties of the output state
    # All amplitudes should be 1/sqrt(N) = 0.5
    ASSERT All_Close(amplitudes, [0.5, 0.5, 0.5, 0.5])
    # Phases should show a linear progression
    ASSERT All_Close(phases, [0, PI/2, PI, -PI/2] or [0, PI/2, PI, 3*PI/2])
    # The state must be normalized
    ASSERT Is_Normalized(final_state_vector)

    RETURN final_state_vector, amplitudes, phases
END FUNCTION
```

#### **Extended Algorithm Sketch (optional)**
```text
Input: |j⟩ with j∈{0,1,2,3}
1) For qubit q0 (LSB): apply H; for m=2..n apply controlled-R_m from higher qubits
2) Repeat for q1 (next bit), skipping already-applied controls; finally swap for bit-reversal
Output: QFT_N|j⟩ with phases ω_N^{jk}
```

#### **Outcome and Interpretation**
The output exhibits equal magnitudes and a linear phase ramp; measuring yields each basis state with probability 1/4, but relative phases enable interference in downstream routines.

---

## **5.2 Quantum Phase Estimation (QPE)** {.heading-with-pill}

> **Difficulty:** ★★★☆☆
> 
> **Concept:** Binary phase readout via controlled powers and inverse QFT
> 
> **Summary:** For $U|\psi\rangle=e^{2\pi i\phi}|\psi\rangle$, QPE estimates $\phi$ to $n$ bits by applying controlled-$U^{2^j}$ from a counting register prepared by Hadamards and concluding with an inverse QFT and measurement.

### **Theoretical Background**

Quantum Phase Estimation (QPE) is the cornerstone algorithm for extracting eigenphases from unitary operators, enabling applications from Shor's algorithm to quantum chemistry simulation.

**Problem Statement:**  
Given a unitary operator $U$ and one of its eigenstates $|\psi\rangle$ satisfying:

$$
U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle
$$

where the eigenphase $\phi \in [0,1)$ is unknown, estimate $\phi$ to $n$ bits of precision.

**Circuit Architecture:**  
Initialize two registers:
- **Counting register:** $n$ qubits in state $|0\rangle^{\otimes n}$  
- **Target register:** eigenstate $|\psi\rangle$

The complete state is $|\psi_0\rangle = |0\rangle^{\otimes n} \otimes |\psi\rangle$.

**Step 1: Hadamard Preparation:**  
Apply Hadamard gates to the counting register:

$$
|\psi_1\rangle = (\mathbf{H}^{\otimes n} \otimes \mathbf{I})|\psi_0\rangle = \frac{1}{2^{n/2}}\sum_{k=0}^{2^n-1} |k\rangle \otimes |\psi\rangle
$$

**Step 2: Controlled Unitary Powers:**  
For each counting qubit $j \in \{0, 1, \ldots, n-1\}$, apply controlled-$U^{2^j}$ with qubit $j$ as control:

$$
|\psi_2\rangle = \frac{1}{2^{n/2}}\sum_{k=0}^{2^n-1} U^k|\psi\rangle \otimes |k\rangle = \frac{1}{2^{n/2}}\sum_{k=0}^{2^n-1} e^{2\pi i k\phi}|k\rangle \otimes |\psi\rangle
$$

Here we used the eigenvalue property: $U^k|\psi\rangle = (e^{2\pi i\phi})^k|\psi\rangle = e^{2\pi ik\phi}|\psi\rangle$.

Note the target register factors out—it remains in $|\psi\rangle$ throughout, unentangled with the counting register.

**Step 3: Inverse QFT:**  
Apply $\text{QFT}_N^{-1}$ to the counting register. Using the inverse transform:

$$
\text{QFT}_N^{-1}|k\rangle = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} e^{-2\pi ijk/N}|j\rangle
$$

The state becomes:

$$
|\psi_3\rangle = \frac{1}{N}\sum_{k=0}^{N-1}\sum_{j=0}^{N-1} e^{2\pi ik(\phi - j/N)}|j\rangle \otimes |\psi\rangle
$$

**Amplitude Analysis:**  
The amplitude of computational basis state $|j\rangle$ is:

$$
\alpha_j = \frac{1}{N}\sum_{k=0}^{N-1} e^{2\pi ik(\phi - j/N)}
$$

This is a geometric series. When $\phi = j/N$ exactly (i.e., $\phi$ has an exact $n$-bit binary representation):

$$
\alpha_j = \begin{cases} 1 & \text{if } j = \lfloor N\phi \rfloor \\ 0 & \text{otherwise} \end{cases}
$$

Measurement deterministically yields $j = \lfloor 2^n \phi \rfloor$, from which $\phi = j/2^n$ exactly.

**Inexact Phase Case:**  
When $\phi$ cannot be exactly represented in $n$ bits, let $\phi = \phi_0 + \delta$ where $\phi_0 = b/2^n$ is the closest $n$-bit approximation and $|\delta| \leq 1/2^{n+1}$. The probability of measuring the best approximation $b$ is:

$$
P(b) = \left|\frac{1}{N}\sum_{k=0}^{N-1} e^{2\pi ik\delta}\right|^2 = \left|\frac{1-e^{2\pi iN\delta}}{N(1-e^{2\pi i\delta})}\right|^2 \geq \frac{4}{\pi^2} \approx 0.405
$$

Thus QPE succeeds with probability $> 40\%$ even for inexact phases, and repeating $\mathcal{O}(1)$ times yields arbitrarily high confidence.

### **Comprehension Check**

!!! note "Quiz"
    **1. Which operation prepares the counting register?**

    - A. $\mathrm{QFT}^{-1}$  
    - B. Hadamards  
    - C. SWAPs  
    - D. Controlled-NOTs  

    ??? info "See Answer"
        **Correct: B.** $\mathbf{H}^{\otimes n}$ creates the uniform superposition.

    **2. Why use powers $U^{2^j}$?**

    - A. To reduce depth  
    - B. To linearize the phase  
    - C. To encode successive binary digits of $\phi$  
    - D. To avoid entanglement  

    ??? info "See Answer"
        **Correct: C.** Exponential powers map bits of $\phi$ into separable phase patterns decodable by $\mathrm{QFT}^{-1}$.

-----

!!! abstract "Interview-Style Question"

    **Q:** What happens when $|\psi\rangle$ is not an eigenstate of $U$?

    ???+ info "Answer Strategy"
        **Eigenstate Decomposition:**  
        QPE extracts eigenphase $\phi$ when the target is an eigenstate $U|\psi_\phi\rangle = e^{2\pi i \phi}|\psi_\phi\rangle$. For non-eigenstates, decompose as $|\psi\rangle = \sum_j c_j |\psi_j\rangle$ where $|\psi_j\rangle$ are eigenstates with eigenvalues $e^{2\pi i \phi_j}$.
        
        $$
        |\psi\rangle = \sum_j c_j |\psi_j\rangle \quad \xrightarrow{\text{QPE}} \quad \sum_j c_j |\tilde{\phi}_j\rangle \otimes |\psi_j\rangle
        $$

        **Probabilistic Outcome:**  
        Measurement yields phase estimate $|\tilde{\phi}_j\rangle$ with probability $P(\phi_j) = |c_j|^2 = |\langle \psi_j|\psi\rangle|^2$, the overlap with eigenstate $|\psi_j\rangle$. This produces a random sample from the eigenphase distribution weighted by state projections.

        **Practical Consequences:**  
        Repeated runs yield different phases with probabilities $|c_j|^2$, sampling the eigenspectrum $\{(\phi_j, |c_j|^2)\}$. In Shor's algorithm, $|1\rangle$ decomposes over eigenstates of $U_a$, sampling uniformly from $\{k/r : k = 0, 1, \ldots, r-1\}$. Continued fractions extract period $r$ from sampled $k/r$. Post-selection on target measurements can prepare specific eigenstates with success probability $|c_k|^2$.

-----

### **<i class="fa-solid fa-flask"></i> Hands-On Projects**

#### **Project Blueprint**
| **Section**              | **Description** |
| ------------------------ | --------------- |
| **Objective**            | Estimate $\phi$ to $n$ bits using QPE with oracle $U$. |
| **Mathematical Concept** | Phase kickback; inverse QFT decoding. |
| **Experiment Setup**     | Choose an eigenpair $(\|\psi\rangle,\phi)$; $n$ counting qubits. |
| **Process Steps**        | Prepare registers; apply controlled-$U^{2^j}$; apply $\mathrm{QFT}^{-1}$; measure. |
| **Expected Behavior**    | Output bitstring approximating $\phi$ in binary. |
| **Tracking Variables**   | Estimated bits; success probability vs precision. |
| **Verification Goal**    | Compare measured estimate to true $\phi$ within $\pm 2^{-n}$. |
| **Output**               | Estimated phase and error bound. |

#### **Pseudocode Implementation**
```pseudo-code
FUNCTION Run_Quantum_Phase_Estimation(U, psi_eigenstate, n_precision_bits):
    # U: The unitary operator whose eigenphase we want to estimate
    # psi_eigenstate: An eigenstate of U, U|psi> = e^(2πiφ)|psi>
    # n_precision_bits: The number of bits of precision for the phase φ
    ASSERT Is_Eigenstate(psi_eigenstate, U)

    # Step 1: Initialize registers
    # n counting qubits in state |0> and a target register with |psi>
    counting_register = Initialize_Qubits(n_precision_bits, state=0)
    initial_state = Tensor_Product(counting_register, psi_eigenstate)
    LOG "Initialized ", n_precision_bits, " counting qubits and target register."

    # Step 2: Apply Hadamard transform to the counting register
    state_after_H = Apply_Hadamard_To_Register(initial_state, register_index=0)
    LOG "Applied H-gates to counting register."

    # Step 3: Apply controlled-U operations
    # For each counting qubit j, apply U^(2^j) controlled by that qubit
    current_state = state_after_H
    FOR j FROM 0 TO n_precision_bits - 1:
        controlled_U_power = Controlled_Unitary(U, power=2**j, control_qubit=j)
        current_state = Apply_Gate(current_state, controlled_U_power)
        LOG "Applied controlled-U^", 2**j, " on control qubit ", j
    END FOR
    state_after_controlled_U = current_state

    # Step 4: Apply the inverse Quantum Fourier Transform (QFT^-1)
    # This transforms the phases into a computational basis state
    final_counting_register = Apply_Inverse_QFT(state_after_controlled_U, register_index=0)
    LOG "Applied Inverse QFT to counting register."

    # Step 5: Measure the counting register
    measured_integer = Measure(final_counting_register, register_index=0)
    LOG "Measured integer value: ", measured_integer

    # Step 6: Convert the integer to the estimated phase
    # φ ≈ measured_integer / 2^n
    estimated_phase = measured_integer / (2**n_precision_bits)
    LOG "Estimated phase φ: ", estimated_phase

    # Verify the estimate is close to the true phase
    true_phase = Get_True_Phase(U, psi_eigenstate)
    ASSERT abs(estimated_phase - true_phase) <= 1 / (2**n_precision_bits)

    RETURN estimated_phase
END FUNCTION
```

#### **Extended Algorithm Sketch (optional)**
```text
Input: U, eigenstate |ψ⟩, precision n
1) Put counting register into uniform superposition via H^{⊗n}
2) For j from 0..n-1 apply controlled-U^{2^j} with control on qubit j
3) Apply inverse QFT on counting register; measure to obtain n-bit estimate of ϕ
```

#### **Outcome and Interpretation**
Controlled powers encode binary digits of the eigenphase; inverse QFT concentrates probability mass on the closest $n$-bit approximation.

---

## **5.3 Phase Kickback** {.heading-with-pill}

> **Difficulty:** ★★☆☆☆
> 
> **Concept:** Converting target phases into control-register phases
> 
> **Summary:** Preparing the control in $|{-}\rangle$ causes a controlled-$U$ to imprint the target’s eigenphase onto the control as a relative phase, enabling interference-based readout.

### **Theoretical Background**

Phase kickback is the quantum phenomenon where a phase factor from the target system of a controlled operation transfers to the control system, enabling indirect measurement of eigenphases without disturbing the eigenstate.

**Mathematical Mechanism:**  
Consider a controlled-unitary gate $\text{ctrl-}U$ acting on control qubit $c$ and target state $|\psi\rangle$ where $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$:

$$
\text{ctrl-}U = |0\rangle\langle 0|_c \otimes \mathbf{I} + |1\rangle\langle 1|_c \otimes U
$$

When the control is in computational basis:

$$
\text{ctrl-}U\big(|0\rangle_c \otimes |\psi\rangle\big) = |0\rangle_c \otimes |\psi\rangle
$$
$$
\text{ctrl-}U\big(|1\rangle_c \otimes |\psi\rangle\big) = |1\rangle_c \otimes U|\psi\rangle = e^{2\pi i\phi}|1\rangle_c \otimes |\psi\rangle
$$

The phase appears as a global factor multiplying the entire state, not directly observable.

**Kickback via Superposition:**  
Prepare the control in $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:

$$
\text{ctrl-}U\left(\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)_c \otimes |\psi\rangle\right) = \frac{1}{\sqrt{2}}\Big(|0\rangle_c \otimes |\psi\rangle + e^{2\pi i\phi}|1\rangle_c \otimes |\psi\rangle\Big)
$$

Factoring out the eigenstate:

$$
= \frac{1}{\sqrt{2}}\big(|0\rangle_c + e^{2\pi i\phi}|1\rangle_c\big) \otimes |\psi\rangle
$$

The eigenphase now appears as a **relative phase** between control qubit basis states, while the target remains in $|\psi\rangle$ (separable, not entangled).

**Bloch Sphere Interpretation:**  
The control qubit state $\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i\phi}|1\rangle)$ represents rotation about the $z$-axis by angle $2\pi\phi$:

$$
|\phi_{\text{ctrl}}\rangle = R_z(2\pi\phi)|+\rangle = e^{-i\pi\phi Z}|+\rangle
$$

**Interferometric Readout:**  
Applying Hadamard to convert relative phase to amplitude:

$$
\mathbf{H}\left(\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i\phi}|1\rangle)\right) = \frac{1}{2}\Big[(1+e^{2\pi i\phi})|0\rangle + (1-e^{2\pi i\phi})|1\rangle\Big]
$$

Simplifying using Euler's formula $e^{2\pi i\phi} = \cos(2\pi\phi) + i\sin(2\pi\phi)$:

$$
= \cos(\pi\phi)|0\rangle + i\sin(\pi\phi)|1\rangle
$$

Measurement probabilities become:

$$
P(0) = \cos^2(\pi\phi), \quad P(1) = \sin^2(\pi\phi)
$$

enabling phase estimation from single-qubit statistics.

**Multi-Qubit Extension (QPE):**  
With $n$ control qubits and controlled-$U^{2^j}$ operations, each control $j$ experiences phase $2^j \phi$:

$$
\frac{1}{2^{n/2}}\sum_{k=0}^{2^n-1} e^{2\pi ik\phi}|k\rangle \otimes |\psi\rangle
$$

Inverse QFT on controls decodes binary digits of $\phi$ while target remains factored in eigenstate $|\psi\rangle$—the essence of quantum phase estimation.

### **Comprehension Check**

!!! note "Quiz"
    **1. Which control-state is convenient for kickback?**

    - A. $|0\rangle$  
    - B. $|1\rangle$  
    - C. $|+\rangle$ or $|{-}\rangle$  
    - D. $|i\rangle$  

    ??? info "See Answer"
        **Correct: C.** Superposition states convert unitary action to relative phase.

    **2. After kickback, how is phase read out?**

    - A. Direct measurement  
    - B. Apply Hadamard(s) to map phase to amplitudes  
    - C. Apply SWAPs  
    - D. Use CZ  

    ??? info "See Answer"
        **Correct: B.** Interference via Hadamards converts phase information into measurable probabilities.

-----

!!! abstract "Interview-Style Question"

    **Q:** Why is kickback critical for reducing multi-qubit eigenphase information to single-qubit interference?

    ???+ info "Answer Strategy"
        **The Information Localization Problem:**  
        Phase $\phi$ from eigenvalue $e^{2\pi i \phi}$ is a global property of multi-qubit state $|\psi\rangle$—directly unmeasurable. Phase kickback transduces this into an observable quantity in a control register.
        
        $$
        \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |\psi\rangle \xrightarrow{\text{ctrl-}U} \frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \phi}|1\rangle) \otimes |\psi\rangle
        $$

        **Single-Qubit Interference:**  
        Eigenphase kicks back to control qubit's relative phase: $|\phi_{\text{ctrl}}\rangle = R_z(2\pi\phi) \cdot |+\rangle$. Applying Hadamard yields measurement probabilities $P(0) = \cos^2(\pi\phi)$, enabling single-qubit interferometric phase estimation. The target register remains unchanged and factored out.

        **Collective Decoding via QFT:**  
        In QPE with $n$ controls, each experiences controlled-$U^{2^j}$, creating $\frac{1}{\sqrt{2^n}}\sum_k e^{2\pi i k\phi}|k\rangle \otimes |\psi\rangle$. Target register is unentangled. Inverse QFT acts as matched filter, concentrating probability into $|\tilde{\phi}\rangle$, the $n$-bit binary representation of $\phi$, achieving exponential precision with polynomial gates.
        
        **Key Advantage:**  
        Without kickback, controlled operations entangle control and target, destroying clean interference. Kickback preserves separability, enabling control-only interferometry while target stores eigenstate unchanged—the architectural principle underlying quantum phase estimation algorithms.

-----

### **<i class="fa-solid fa-flask"></i> Hands-On Projects**

#### **Project Blueprint**
| **Section**              | **Description** |
| ------------------------ | --------------- |
| **Objective**            | Demonstrate single-qubit phase kickback using a controlled-phase gate. |
| **Mathematical Concept** | Relative phase accumulation; Hadamard interferometry. |
| **Experiment Setup**     | Control qubit initialized to $|+\rangle$; target an eigenstate of $U$. |
| **Process Steps**        | Apply controlled-$U$; then Hadamard on control; measure. |
| **Expected Behavior**    | Outcome probabilities depend on $\phi$. |
| **Tracking Variables**   | Control measurement $P(0),P(1)$ as functions of $\phi$. |
| **Verification Goal**    | Match $P(0)=\cos^2(\pi\phi)$, $P(1)=\sin^2(\pi\phi)$. |
| **Output**               | Estimated $\phi$ from observed counts. |

#### **Pseudocode Implementation**
```pseudo-code
FUNCTION Demonstrate_Phase_Kickback(U, psi_eigenstate):
    # U: A unitary with eigenstate |psi> and eigenphase φ
    # psi_eigenstate: The eigenstate U|psi> = e^(2πiφ)|psi>
    ASSERT Is_Eigenstate(psi_eigenstate, U)

    # Step 1: Prepare the initial state
    # Control qubit in |+> state, target register in |psi>
    control_qubit = (1/sqrt(2)) * (State_0() + State_1())
    initial_state = Tensor_Product(control_qubit, psi_eigenstate)
    LOG "Initial state: |+>|psi>"

    # Step 2: Apply the controlled-U operation
    # This "kicks back" the phase e^(2πiφ) to the control qubit
    # State becomes: 1/sqrt(2) * (|0>|psi> + e^(2πiφ)|1>|psi>)
    state_after_kickback = Apply_Controlled_Unitary(initial_state, U, control_qubit_index=0)
    LOG "State after kickback: Phase φ transferred to control qubit."

    # Step 3: Apply Hadamard gate to the control qubit
    # This converts the relative phase into measurable amplitude differences
    # Final control state: cos(πφ)|0> + i*sin(πφ)|1>
    final_state = Apply_Hadamard(state_after_kickback, qubit_index=0)
    LOG "Applied Hadamard to control qubit for interference."

    # Step 4: Measure the control qubit
    # Probabilities depend on the kicked-back phase φ
    measurement_probabilities = Compute_Probabilities(final_state, register_index=0)
    P0 = measurement_probabilities[0] # Probability of measuring |0>
    P1 = measurement_probabilities[1] # Probability of measuring |1>
    LOG "P(0) = ", P0, ", P(1) = ", P1

    # Step 5: Verify the result
    true_phase = Get_True_Phase(U, psi_eigenstate)
    expected_P0 = cos(PI * true_phase)**2
    expected_P1 = sin(PI * true_phase)**2
    ASSERT abs(P0 - expected_P0) < 1e-9
    ASSERT abs(P1 - expected_P1) < 1e-9
    LOG "Probabilities match theoretical values."

    RETURN P0, P1
END FUNCTION
```

#### **Extended Algorithm Sketch (optional)**
```text
1) Put control in |+⟩; target as eigenstate |ψ⟩
2) Apply controlled-U; relative phase e^{2πiϕ} appears on |1⟩ component
3) Apply H to map phase to amplitudes; measure control
```

#### **Outcome and Interpretation**
You convert hidden eigenphase into measurable bias on a single qubit, the primitive underpinning QPE.

---

## **5.4 Order Finding and Applications** {.heading-with-pill}

> **Difficulty:** ★★★☆☆
> 
> **Concept:** Period estimation in modular arithmetic via QPE
> 
> **Summary:** For $f(x)=a^x\bmod N$, QPE over the unitary $U_a:|x\rangle\mapsto|ax\bmod N\rangle$ estimates the order $r$ where $a^r\equiv1\pmod N$; classical post-processing via gcd yields nontrivial factors.

### **Theoretical Background**

Order finding is the quantum subroutine underlying Shor's factoring algorithm, reducing the problem of finding the multiplicative order of an element in modular arithmetic to quantum phase estimation.

**Problem Definition:**  
Given integers $a$ and $N$ with $\gcd(a,N)=1$, find the smallest positive integer $r$ (the order or period) such that:

$$
a^r \equiv 1 \pmod{N}
$$

**Modular Multiplication as Unitary:**  
Define the unitary operator $U_a$ acting on the computational basis of $\mathbb{Z}_N = \{0, 1, \ldots, N-1\}$:

$$
U_a|y\rangle = |ay \bmod N\rangle
$$

This is unitary since multiplication by $a \bmod N$ is a permutation (bijection) when $\gcd(a,N)=1$. Controlled powers can be implemented efficiently:

$$
U_a^{2^j}|y\rangle = |a^{2^j} y \bmod N\rangle
$$

using $\mathcal{O}(\log^2 N)$ gates via repeated squaring.

**Eigenvalue Structure:**  
The eigenstates of $U_a$ are:

$$
|\psi_s\rangle = \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1} e^{-2\pi isk/r}|a^k \bmod N\rangle \quad \text{for } s \in \{0, 1, \ldots, r-1\}
$$

Verifying the eigenvalue property:

$$
U_a|\psi_s\rangle = \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1} e^{-2\pi isk/r}|a^{k+1} \bmod N\rangle = \frac{1}{\sqrt{r}}\sum_{\ell=1}^{r} e^{-2\pi is(\ell-1)/r}|a^\ell \bmod N\rangle
$$

Substituting $\ell-1 \to k$ and using periodicity $a^r \equiv 1$:

$$
= e^{2\pi is/r} \cdot \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1} e^{-2\pi isk/r}|a^k \bmod N\rangle = e^{2\pi is/r}|\psi_s\rangle
$$

Thus the eigenvalues are $\lambda_s = e^{2\pi is/r}$ with eigenphases:

$$
\phi_s = \frac{s}{r} \quad \text{for } s \in \{0, 1, \ldots, r-1\}
$$

**QPE Application:**  
Starting with $|1\rangle$ (not an eigenstate), decompose:

$$
|1\rangle = \frac{1}{\sqrt{r}}\sum_{s=0}^{r-1} |\psi_s\rangle
$$

Applying QPE produces:

$$
\frac{1}{\sqrt{r}}\sum_{s=0}^{r-1} |\tilde{\phi}_s\rangle \otimes |\psi_s\rangle
$$

Measuring the phase register yields random $s \in \{0, \ldots, r-1\}$ with equal probability $1/r$, providing estimate $\tilde{\phi} \approx s/r$.

**Continued Fractions Extraction:**  
From measured approximation $m/2^n \approx s/r$, the continued fractions algorithm finds the fraction with smallest denominator within precision:

$$
\left|\frac{s}{r} - \frac{m}{2^n}\right| \leq \frac{1}{2^{n+1}}
$$

Choosing $n = 2\log_2 N$ ensures $1/2^{n+1} < 1/(2N^2) < 1/(2r^2)$, guaranteeing $s/r$ appears as a convergent.

**Factor Extraction:**  
Once $r$ is determined, if:
1. $r$ is even, and  
2. $a^{r/2} \not\equiv -1 \pmod{N}$

then $a^r - 1 = (a^{r/2}-1)(a^{r/2}+1) \equiv 0 \pmod{N}$, but $N$ divides the product without dividing individual factors. Computing:

$$
\gcd(a^{r/2} \pm 1, N)
$$

yields nontrivial factors with probability $\geq 1/2$ over random choice of $a$.

### **Comprehension Check**

!!! note "Quiz"
    **1. Order finding seeks:**

    - A. Minimal $r$ with $a^r\equiv1\pmod N$  
    - B. Minimal $r$ with $a^r\equiv0\pmod N$  
    - C. $\gcd(a,N)$  
    - D. A discrete log  

    ??? info "See Answer"
        **Correct: A.** The multiplicative order.

    **2. The classical finishing step uses:**

    - A. Euclid’s algorithm  
    - B. FFT  
    - C. Gradient descent  
    - D. Simulated annealing  

    ??? info "See Answer"
        **Correct: A.** $\gcd$ computations extract factors from $a^{r/2}\pm1$.

-----

!!! abstract "Interview-Style Question"

    **Q:** Why is continued-fraction decoding necessary after QPE in order finding?

    ???+ info "Answer Strategy"
        **The Rational Approximation Problem:**  
        QPE estimates eigenphase $\phi = k/r$ to $n$ bits, yielding measurement $m \approx 2^n \cdot k/r$. Direct division $m/2^n$ gives decimal approximation like $0.748031...$ but doesn't reveal underlying fraction $k/r$. Naively rounding is unreliable among exponentially many possible fractions.
        
        $$
        \left|\frac{k}{r} - \frac{m}{2^n}\right| \leq \frac{1}{2^{n+1}} < \frac{1}{2r^2}
        $$

        **Continued Fractions Algorithm:**  
        Continued fraction expansion produces convergents $p_i/q_i$ that are best rational approximations. Fundamental theorem: if $|x - k/r| \leq 1/(2r^2)$, then $k/r$ appears as a convergent. For QPE output $m/2^n$, compute convergents, check $q_i \leq N$ and verify $a^{q_i} \equiv 1 \pmod{N}$ to find order $r$.

        **Robustness and Efficiency:**  
        Continued fractions automatically find simplest fraction closest to measured value in $\mathcal{O}(\log N)$ operations. Handles measurement errors gracefully—even if $m$ is off by units, correct $k/r$ remains a convergent. Example: $N=15$, $a=2$, $r=4$. QPE yields $m=64$, giving $1/4$ with convergent denominator $q=4$. Checking $2^4 \equiv 1 \pmod{15}$ confirms $r=4$.
        
        **Why Alternatives Fail:**  
        Brute force testing all $r \leq N$ requires $\mathcal{O}(N)$ complexity, eliminating quantum advantage. Rounding heuristics fail for large $r$ or common factors. Continued fractions uniquely extract denominators efficiently from finite-precision approximations.

-----

### **<i class="fa-solid fa-flask"></i> Hands-On Projects**

#### **Project Blueprint**
| **Section**              | **Description** |
| ------------------------ | --------------- |
| **Objective**            | Outline the QPE-based order-finding workflow and its classical post-processing. |
| **Mathematical Concept** | Eigenphases $k/r$; continued fractions; gcd extraction. |
| **Experiment Setup**     | Choose small $N$ (e.g., 15) and $a$ coprime to $N$. |
| **Process Steps**        | Run conceptual QPE to estimate $k/r$; apply continued fractions; compute $\gcd(a^{r/2}\pm1,N)$. |
| **Expected Behavior**    | Recover $r$ and nontrivial factors. |
| **Tracking Variables**   | Phase estimates; convergents; gcd outputs. |
| **Verification Goal**    | Confirm factors multiply to $N$. |
| **Output**               | $r$ and factors of $N$. |

#### **Pseudocode Implementation**
```pseudo-code
FUNCTION Find_Order_Of_a_Mod_N(a, N):
    # a: An integer coprime to N
    # N: The modulus
    ASSERT Is_Coprime(a, N)

    # Step 1: Quantum Phase Estimation to find the period 'r'
    # Define the unitary U|x> = |a*x mod N>
    U_a = Create_Modular_Multiplication_Unitary(a, N)
    
    # QPE requires an eigenstate. A superposition of all eigenstates works.
    # We can start with |1> which can be decomposed into eigenstates.
    initial_target_state = State_Vector_For_Integer(1, num_qubits_for_N)
    
    # Run QPE to get an estimate of s/r for some integer s
    # The number of precision bits determines the accuracy
    n_precision = 2 * log2(N) + 1 # Recommended precision
    phase_estimate = Run_Quantum_Phase_Estimation(U_a, initial_target_state, n_precision)
    LOG "QPE phase estimate (s/r): ", phase_estimate

    # Step 2: Use Continued Fractions algorithm to find r
    # This classical algorithm finds the best rational approximation s/r
    s, r = Continued_Fractions(phase_estimate, max_denominator=N)
    LOG "Recovered period r: ", r

    # Step 3: Verify the period
    # Check if a^r ≡ 1 (mod N)
    IF Power(a, r, N) != 1:
        RETURN "Failure: Could not find correct period."
    
    # Step 4: Use the period to find factors of N (Shor's algorithm part)
    IF r % 2 == 0:
        y = Power(a, r/2, N)
        IF y != N - 1:
            factor1 = GCD(y - 1, N)
            factor2 = GCD(y + 1, N)
            IF factor1 > 1 OR factor2 > 1:
                LOG "Found factors: ", factor1, factor2
                RETURN r, [factor1, factor2]
            END IF
        END IF
    END IF

    RETURN r, "No non-trivial factors found from this 'a'."
END FUNCTION
```

#### **Extended Algorithm Sketch (optional)**
```text
1) Prepare eigenstate superposition of U_a and apply QPE to get phase ≈ k/r
2) Use continued fractions to recover r from phase estimate
3) If r is even and a^{r/2}≢−1 (mod N), compute gcd(a^{r/2}±1,N)
```

#### **Outcome and Interpretation**
This workflow isolates the quantum advantage (phase estimation) from classical number-theoretic post-processing.

---

## **5.5 Approximate QFT and Depth Trade-Offs** {.heading-with-pill}

> **Difficulty:** ★★☆☆☆
> 
> **Concept:** Truncating small-angle rotations to reduce depth
> 
> **Summary:** Dropping controlled rotations below a threshold yields an approximate QFT with lower depth and gate count; the induced error can be bounded and is often acceptable on NISQ devices.

### **Theoretical Background**

The approximate QFT reduces circuit depth by truncating small-angle controlled rotations while maintaining sufficient accuracy for phase estimation applications on NISQ hardware.

**Exact QFT Gate Structure:**  
The standard QFT on qubit $j$ requires controlled rotations from all higher qubits $k > j$:

$$
\mathbf{H}_j \prod_{k=j+1}^{n-1} R_{k-j+1}^{(k,j)} \quad \text{where} \quad R_m = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i/2^m} \end{pmatrix}
$$

For large separation $k-j$, the rotation angle $\theta_m = 2\pi/2^m$ becomes exponentially small. For example:
- $m=10$: $\theta_{10} = 2\pi/1024 \approx 0.0061$ rad  
- $m=15$: $\theta_{15} = 2\pi/32768 \approx 0.00019$ rad  
- $m=20$: $\theta_{20} = 2\pi/1048576 \approx 6 \times 10^{-6}$ rad

**Truncation Strategy:**  
Omit all rotations $R_m$ with $m > t$ for some threshold $t$. The approximate QFT becomes:

$$
\text{QFT}_{\text{approx}} = \text{SWAP}_{\text{reverse}} \cdot \left(\prod_{j=n-1}^{0} \mathbf{H}_j \prod_{k=j+1}^{\min(j+t, n-1)} R_{k-j+1}^{(k,j)}\right)
$$

This reduces the number of controlled rotations from $\frac{n(n-1)}{2}$ to approximately $nt$.

**Error Analysis:**  
Define the fidelity between exact and approximate QFT acting on state $|\psi\rangle$:

$$
F = |\langle\psi|\text{QFT}^\dagger_{\text{approx}} \text{QFT}_{\text{exact}}|\psi\rangle|^2
$$

For uniformly random input states, the average fidelity is bounded by:

$$
\mathbb{E}[F] \geq 1 - \epsilon \quad \text{where} \quad \epsilon \lesssim \frac{n}{2^t}
$$

Thus choosing $t = \log_2(n) + b$ yields error $\epsilon \sim 2^{-b}$.

**Phase Error in QPE:**  
The induced phase error in quantum phase estimation is:

$$
|\delta\phi| \lesssim \frac{1}{2^t}
$$

Since QPE needs precision $1/2^n$, the approximation is acceptable when:

$$
t \geq n - \log_2(n) - c
$$

for small constant $c$, typically $c \approx 3-5$.

**Circuit Depth Reduction:**  
Exact QFT has depth $\mathcal{O}(n^2)$ due to $n$ layers each with $\mathcal{O}(n)$ gates. Approximate QFT reduces depth to:

$$
D_{\text{approx}} \approx n + t \cdot \log_2(\text{connectivity})
$$

For $t = \mathcal{O}(\log n)$, this becomes $\mathcal{O}(n \log n)$, a quadratic improvement.

**NISQ Hardware Considerations:**  
On devices with gate error rate $\epsilon_{\text{gate}} \sim 10^{-3}$ and coherence time limiting depth to $D_{\max} \sim 100-200$ gates, choose truncation level where algorithmic error matches hardware noise:

$$
2^{-t} \sim \epsilon_{\text{gate}} \cdot D_{\text{approx}} \implies t \approx \log_2\left(\frac{1}{\epsilon_{\text{gate}} \cdot n}\right)
$$

Typically $t \in [8, 15]$ for $n \in [10, 25]$ qubits on current NISQ devices.

### **Comprehension Check**

!!! note "Quiz"
    **1. Approximating QFT mainly reduces:**

    - A. Width  
    - B. Depth and two-qubit gates  
    - C. Classical post-processing  
    - D. Measurement shots  

    ??? info "See Answer"
        **Correct: B.** Small-angle rotations are costly and sensitive to noise.

    **2. The trade-off of approximation is:**

    - A. No change  
    - B. Bounded phase error  
    - C. Fewer qubits  
    - D. Exact results  

    ??? info "See Answer"
        **Correct: B.** Errors grow with dropped angles but can be analytically bounded.

-----

!!! abstract "Interview-Style Question"

    **Q:** When would you prefer an approximate QFT in QPE on NISQ hardware?

    ???+ info "Answer Strategy"
        **The Depth-Precision Trade-off:**  
        Exact QFT requires $\mathcal{O}(n^2)$ gates including controlled rotations $R_k(\theta)$ with angles $\theta_j = 2\pi/2^j$. For large $j$, rotations become tiny ($\theta_{15} \approx 0.0002$ rad). On NISQ devices with gate errors $\epsilon_{\text{gate}} \sim 10^{-3}$ and coherence $T_2 \sim 50-500~\mu$s, these small rotations are overwhelmed by noise.
        
        $$
        |\delta\phi| \lesssim \frac{1}{2^m}
        $$

        **When to Use Approximate QFT:**  
        Omit rotations with $j > m$ when noise magnitude exceeds signal. For Shor's algorithm factoring $N \sim 2048$ needing $\sim 22$ bits, exact QFT requires $\sim 484$ gates exceeding typical depth budget of $\sim 200$ gates. Approximate QFT truncating to $m = 15$ uses $\sim 225$ gates, fitting within coherence limits while providing sufficient $1/2^{15}$ precision for continued fractions.

        **Circuit Depth Reduction:**  
        Approximate QFT reduces gate count from $\mathcal{O}(n^2)$ to $\mathcal{O}(nm)$ and depth proportionally. Example: $n=20$, $m=10$ reduces gates from $\sim 400$ to $\sim 200$ and depth from $\sim 190$ to $\sim 95$. Choose $m$ where $1/2^m \sim \epsilon_{\text{hw}}$ (matching noise floor) and $1/2^m < \epsilon_{\text{app}}$ (meeting application requirements).
        
        **Practical Decision:**  
        On IBM Quantum with $\epsilon_{\text{hw}} \sim 10^{-3}$ and depth budget $\sim 150$ gates, use $m = 12$ yielding precision $1/2^{12} \approx 2.4 \times 10^{-4}$ and depth $\sim 132$ gates. Exact QFT would require $\sim 288$ gates, guaranteed failure. On fault-tolerant devices with $\epsilon_{\text{gate}} < 10^{-10}$, use exact QFT for maximum precision.

-----

### **<i class="fa-solid fa-flask"></i> Hands-On Projects**

#### **Project Blueprint**
| **Section**              | **Description** |
| ------------------------ | --------------- |
| **Objective**            | Compare depth and expected phase error between exact and approximate QFT for $n$ qubits. |
| **Mathematical Concept** | Thresholding controlled rotations; error vs precision. |
| **Experiment Setup**     | Choose $n\in\{4,6,8\}$; drop rotations below angle $2\pi/2^t$. |
| **Process Steps**        | Count gates and layers for exact vs truncated; estimate phase error bound. |
| **Expected Behavior**    | Significant depth reduction with small increase in error. |
| **Tracking Variables**   | Gate counts, depth, error bound. |
| **Verification Goal**    | Confirm error within application tolerance. |
| **Output**               | Depth/error table and narrative. |

#### **Pseudocode Implementation**
```pseudo-code
FUNCTION Analyze_Approximate_QFT(n_qubits, truncation_level_t):
    # n_qubits: Total number of qubits
    # truncation_level_t: Controls which small-angle rotations are dropped.
    # Rotations R_m where m > t are dropped.
    ASSERT n_qubits > 0 AND truncation_level_t > 0

    # --- Exact QFT Resource Count ---
    # Step 1: Calculate depth and gate count for the exact QFT
    exact_num_H = n_qubits
    exact_num_CR = (n_qubits * (n_qubits - 1)) / 2 # Controlled-Rotations
    exact_depth = 2 * n_qubits # Approximation, depends on connectivity
    LOG "Exact QFT (n=", n_qubits, "):"
    LOG "  Controlled-Rotations: ", exact_num_CR
    LOG "  Depth (approx): ", exact_depth

    # --- Approximate QFT Resource Count ---
    # Step 2: Calculate resources for the approximate QFT
    approx_num_H = n_qubits
    approx_num_CR = 0
    FOR i FROM 1 TO n_qubits:
        # For qubit i, we add controlled rotations from qubit j > i
        # up to the truncation level t
        num_rotations_for_qubit_i = min(n_qubits - i, truncation_level_t - 1)
        approx_num_CR += max(0, num_rotations_for_qubit_i)
    END FOR
    
    # Depth is reduced because fewer gates are applied
    approx_depth = n_qubits + truncation_level_t # Rough approximation
    LOG "Approximate QFT (n=", n_qubits, ", t=", truncation_level_t, "):"
    LOG "  Controlled-Rotations: ", approx_num_CR
    LOG "  Depth (approx): ", approx_depth

    # Step 3: Estimate the error introduced by the approximation
    # The error is bounded and depends on the dropped rotations
    # A known bound on the fidelity F is F >= (1 - ε)^2 where ε is small
    error_bound = Calculate_Fidelity_Error_Bound(n_qubits, truncation_level_t)
    LOG "Estimated Fidelity Error Bound: ", error_bound

    # Step 4: Report the trade-off
    depth_reduction = exact_depth - approx_depth
    gate_reduction = exact_num_CR - approx_num_CR
    LOG "Trade-off Summary:"
    LOG "  Gate reduction: ", gate_reduction
    LOG "  Depth reduction: ", depth_reduction
    LOG "  Introduced error is bounded by: ", error_bound

    RETURN {
        "depth_reduction": depth_reduction,
        "gate_reduction": gate_reduction,
        "error_bound": error_bound
    }
END FUNCTION
```

#### **Extended Algorithm Sketch (optional)**
```text
1) For each qubit, omit controlled-R_m with m>t
2) Recompute layer schedule and two-qubit counts
3) Use known bounds to estimate induced phase error
```

#### **Outcome and Interpretation**
Approximate QFT provides a practical path to phase estimation on noisy hardware by trading a small accuracy loss for large depth savings.
