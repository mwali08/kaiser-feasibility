# Kaiser – Feasibility Study & PoC

This repository serves as the **experimental and testing environment** for the technical feasibility study (*Proof of Concept*) of the diploma thesis project **"Kaiser"**. 

Here, team members can test different technologies, evaluate components, and build functional prototypes before deciding on the final stack to be integrated into the main repository.

---

## 🎯 Purpose of this Repository
- **Technology Evaluation:** Testing and comparing various frameworks, libraries, and tools.
- **Prototyping (PoC):** Building functional prototypes to validate technical feasibility.
- **Pipeline & Infrastructure Tests:** Experimenting with CI/CD pipelines and deployment strategies.

---

## Workflow & Branching Strategy

To keep this repository clean and organized, please follow these rules:

1. **Independent Branches:** Every team member creates their own branches to test specific technologies or features.
   - *Naming convention:* `<type>/<topic-or-technology>` (e.g., `eval/tech-stack-a`, `poc/auth-system`, `sandbox/username`).
2. **Pull Request (PR) Policy:** Direct pushes to the `main` branch are **prohibited**. 
   - All changes, prototypes, and tests must be submitted via a **Pull Request (PR)** to the `main` branch.
   - Every PR should be reviewed or at least documented clearly so the rest of the team understands what was tested and what the results/takeaways are.

---

## Interface Documentation for Aron (Offline Environment & Integration)

Aron is responsible for building the offline environment and connecting the individual modules. Therefore, **every component or prototype tested in this repository must include interface documentation**.

If your prototype or technology choice exposes APIs, data formats, or specific system requirements, please document them directly. 

### Required Information for Aron:
* **Endpoints / APIs:** What routes, methods (GET/POST), or function calls does your component provide?
* **Data Formats:** What do the input and output payloads look like? (Provide a short JSON or schema example if applicable).
* **Dependencies & Prerequisites:** What libraries, runtimes, ports, or environment variables does your part require to run in the offline environment?
* **Communication Protocol:** How does your component communicate (e.g., REST, gRPC, WebSockets, file sharing)?
* **Docker:** Docker File specifying which versions and settings to use. 

---

## Repository Structure
```text
kaiser-feasibility/
│
├── docs/                 # General documentation & interface notes
└── README.md             # This file
