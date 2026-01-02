## Chapter 5.1: Bit-flip and Phase-flip Codes (Workbook)

The goal of this chapter is to introduce the foundational principles of **Quantum Error Correction (QEC)** by analyzing the simplest repetition codes—the **3-qubit bit-flip code** and the **3-qubit phase-flip code**—and explaining the concept of **syndrome extraction** using ancilla qubits.

***

### 5.1.1 Types of Quantum Errors and Classical Analogy

> **Summary:** Quantum errors are not just bit flips ($X$ errors) but also **phase flips** ($Z$ errors). A general error is a combination of Pauli errors ($I, X, Y, Z$). The QEC philosophy is based on the **classical repetition code** (e.g., encoding $0$ as $000$) but must avoid collapsing the protected quantum superposition.

#### Quiz Questions
**1. An error that introduces a minus sign on the $|1\rangle$ component of a superposition state ($\alpha|0\rangle + \beta|1\rangle \to \alpha|0\rangle - \beta|1\rangle$) is known as a:**
* **A.** Bit-flip ($X$ error).
* **B.** **Phase-flip ($Z$ error)**. (Correct)
* **C.** $Y$ error.
* **D.** Ancilla error.

**2. Which theorem dictates that a QEC process cannot simply copy the quantum state multiple times to achieve redundancy?**
* **A.** The Threshold Theorem.
* **B.** The Stabilizer Formalism.
* **C.** **The No-cloning theorem**. (Correct)
* **D.** The Adiabatic Theorem.

#### Interview-Style Question
**Question:** Explain why protecting a logical qubit against a general single-qubit error requires protection against both $X$ and $Z$ errors, and not just $X$ errors.

**Answer Strategy:**
* **Pauli Error Basis:** A general, arbitrary error on a single qubit is mathematically represented as a linear combination of the four Pauli operators ($I, X, Y, Z$).
* **$Y$ Error:** The $Y$ error is the product of an $X$ error and a $Z$ error ($Y = iXZ$).
* **Necessity:** Therefore, any code that claims to correct a **general single-qubit error** must be able to independently detect and correct the two fundamental, non-commuting error components: the **bit-flip ($X$)** and the **phase-flip ($Z$)**. If a code only corrects $X$, it will fail on $Z$ or $Y$ errors.

***

### 5.1.3–5.1.4 Bit-flip and Phase-flip Codes

> **Summary:** The **3-qubit bit-flip code** protects against single $X$ errors by encoding $|{\psi}\rangle = \alpha |0\rangle + \beta |1\rangle$ as $|{\psi}_L\rangle = \alpha |000\rangle + \beta |111\rangle$. The **3-qubit phase-flip code** protects against single $Z$ errors by applying the bit-flip code logic in the Hadamard basis, resulting in the logical state $|{\psi}_L\rangle = \alpha |+++\rangle + \beta |---\rangle$. Both use **stabilizer measurements** to determine the error **syndrome**.

#### Quiz Questions
**1. What is the logical state $|1_L\rangle$ for the 3-qubit bit-flip code?**
* **A.** $|+++\rangle$.
* **B.** $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$.
* **C.** **$|111\rangle$**. (Correct)
* **D.** $|100\rangle$.

**2. The error detection for the 3-qubit bit-flip code relies on measuring the parity of which pairs of qubits?**
* **A.** $X_1 X_2$ and $X_2 X_3$.
* **B.** $X_1 Z_2$ and $Z_2 X_3$.
* **C.** **$Z_1 Z_2$ and $Z_2 Z_3$**. (Correct)
* **D.** $H_1 H_2$ and $H_2 H_3$.

#### Interview-Style Question
**Question:** Explain the intuitive reason why the **Phase-flip code** uses the **Hadamard basis** (states $|+\rangle, |-\rangle$) for its logical states.

**Answer Strategy:**
* **Interchangeability:** The Hadamard gate has the property that it maps the Pauli $X$ operator to the Pauli $Z$ operator, and vice-versa.
* **Rotation:** A **phase-flip ($Z$ error)** in the computational basis is equivalent to a **bit-flip ($X$ error)** in the Hadamard basis.
* **Intuition:** By rotating the code into the Hadamard basis using $H$ gates, the physical $Z$ error effectively becomes a physical $X$ error. The code can then simply reuse the established **Bit-flip code mechanism** (which is easy to analyze using majority voting) to correct the error, and then rotate back.

***

### 5.1.5–5.1.6 Syndrome Measurement and Stabilizers

> **Summary:** The core challenge of QEC is **syndrome extraction**, which detects the error location without collapsing the superposition of the data qubits. This is achieved using **ancilla qubits** entangled with the data qubit parities via CNOT gates. The code states must satisfy the **stabilizer formalism**, meaning all states $|{\psi}\rangle$ are eigenvectors with eigenvalue $+1$ for all stabilizer operators $S_i$.

#### Quiz Questions
**1. The primary purpose of using **ancilla qubits** in the error correction process is to:**
* **A.** Store the logical state redundantly.
* **B.** **Measure the error syndrome without collapsing the data qubit's superposition**. (Correct)
* **C.** Perform the correction by applying an inverse gate.
* **D.** Separate bit-flip and phase-flip errors.

**2. According to the stabilizer formalism, what must be true for all valid code states $|{\psi}\rangle$ and all stabilizer operators $S_i$?**
* **A.** $S_i |{\psi}\rangle = -|{\psi}\rangle$.
* **B.** $S_i |{\psi}\rangle = 0$.
* **C.** **$S_i |{\psi}\rangle = +|{\psi}\rangle$**. (Correct)
* **D.** $S_i |{\psi}\rangle = \delta_{ij} |{\psi}\rangle$.

## Hands-On Workbook Projects

These projects focus on demonstrating the encoding, stabilizer properties, and error detection logic of the 3-qubit codes.

### Project 1: Bit-flip Code Error Correction Logic

* **Goal:** Practice the error identification logic of the 3-qubit bit-flip code.
* **Setup:** The code measures the two stabilizers $S_A = Z_1 Z_2$ and $S_B = Z_2 Z_3$. The measured syndrome is a binary pair $(S_A, S_B)$.
* **Steps:**
    1.  An error $E=X_3$ (bit-flip on qubit 3) occurs. Calculate the resulting syndrome $(S_A, S_B)$. (Hint: $X_3$ commutes with $Z_1 Z_2$, but anti-commutes with $Z_2 Z_3$)
    2.  An error $E=X_2$ (bit-flip on qubit 2) occurs. Calculate the resulting syndrome $(S_A, S_B)$.
    3.  Explain how a decoding algorithm uses the syndrome $(1, 0)$ to determine the correction.

### Project 2: Phase-flip Code Encoding

* **Goal:** Verify the encoding steps for the phase-flip code.
* **Setup:** Start with the logical state $|{\psi}\rangle = \alpha |0\rangle + \beta |1\rangle$. The goal is to reach $|{\psi}_L\rangle = \alpha |+++\rangle + \beta |---\rangle$.
* **Steps:**
    1.  Write the basis states $|+\rangle$ and $|-\rangle$ in the computational basis $\{|0\rangle, |1\rangle\}$.
    2.  Write the explicit computational basis expansion for the target state $|+++\rangle$.
    3.  Explain, without using gates, why the encoding $|0\rangle \to |+++\rangle$ and $|1\rangle \to |---\rangle$ protects against $Z$ errors. (Hint: What does the $Z$ error do to $|+\rangle$ and $|-\rangle$?)

### Project 3: Logical Operator Identity

* **Goal:** Understand the properties of logical operators.
* **Setup:** For the 3-qubit bit-flip code, the logical $X$ operator is defined as $X_L = X_1 X_2 X_3$.
* **Steps:**
    1.  Apply $X_L$ to the logical state $|0_L\rangle = |000\rangle$. Show the result is the logical $|1_L\rangle$.
    2.  Show that $X_L$ commutes with the stabilizer $S_A = Z_1 Z_2$. (Hint: $X$ and $Z$ anti-commute, $[X, Z] = 2iY$).
    3.  Explain why commuting with the stabilizer is a necessary condition for a logical operator.

### Project 4: Error Detection vs. Correction

* **Goal:** Distinguish between error detection and correction.
* **Setup:** Consider a simpler 2-qubit code: $|{\psi}_L\rangle = \alpha |00\rangle + \beta |11\rangle$. Stabilizer is $S = Z_1 Z_2$.
* **Steps:**
    1.  Verify that this code can **detect** an $X_1$ error. (Hint: Check if the syndrome measurement $Z_1 Z_2$ changes from $+1$ to $-1$)
    2.  Explain why this code **cannot correct** the $X_1$ error. (Hint: What would the syndrome be for an $X_2$ error?)


