# Nura — Long-Lived Memory System for Conversational AI

> An engineering effort focused on the systematic design of a memory-centric AI system with explicit architectural constraints and strict separation of concerns.

## Overview

Nura is a four-engine memory architecture designed for long-horizon conversational AI with persistent, adaptive memory capabilities. The system prioritizes architectural discipline and dependency hygiene over rapid feature development.

## Architecture

### Four-Engine Design
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Memory    │  │  Retrieval  │  │  Temporal   │  │ Adaptation  │
│   Engine    │  │   Engine    │  │   Engine    │  │   Engine    │
│             │  │             │  │             │  │             │
│  Storage &  │  │  Semantic   │  │    Time     │  │  Pattern    │
│   Facts     │  │   Search    │  │  Reasoning  │  │  Learning   │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

**Memory Engine** — Event ingestion, classification, persistent storage  
**Retrieval Engine** — Semantic search, ranking, relevance scoring  
**Temporal Engine** — Time awareness, temporal patterns, context generation  
**Adaptation Engine** — User profile evolution, behavioral adaptation  

### Design Principles

- **Strict Separation of Concerns** — Each engine has non-overlapping responsibilities
- **Boundary Enforcement** — No cross-engine imports; responsibilities enforced at module level
- **Test Isolation** — Components testable independently via protocol-based interfaces
- **Architectural Discipline** — Correctness before features; structure before optimization

## Current Status

### Phase 1: Architectural Boundary Enforcement ✅ **COMPLETED**

**Objective:** Eliminate cross-engine dependency violations before activating higher-order behavior.

**Sub-phases:**
- ✅ **1.1 Temporal Tag Extraction** — Removed temporal logic from Memory Engine
- ✅ **1.2 Embedding Generation Extraction** — Isolated embedding responsibility from core engines
- ✅ **1.3 Retrieval–Temporal Boundary Enforcement** — Corrected temporal parsing ownership and removed illegal imports

**Outcome:**
- Cross-engine imports: Eliminated
- Engine responsibility boundaries: Enforced
- Temporal reasoning ownership: Centralized
- Runtime behavior changes: None (by design)

**Status:** Validated and complete. System ready for Phase 2.

---

### Phase 2: Cross-Engine Orchestration 📋 **PLANNED**

**Objective:** Activate existing features through coordinated multi-engine workflows.

**Scope:**
- Adaptation profile integration into response generation
- Temporal pattern detection and utilization
- Temporal-aware retrieval scoring
- Memory summarization implementation

**Design Principle:** Orchestration occurs at higher abstraction layer; core engines remain stateless and independently testable.

**Phase 2 initiation contingent upon Phase 1 validation completion.**

## Project Structure
```
nura/
├── engines/
│   ├── memory/          # Storage, classification, fact extraction
│   ├── retrieval/       # Semantic search, ranking
│   ├── temporal/        # Time reasoning, pattern detection
│   └── adaptation/      # Profile building, effectiveness tracking
├── shared/
│   ├── database.py      # SQLite session management
│   ├── embeddings.py    # Vector embedding service
│   └── config.py        # Centralized configuration
├── tests/               # Unit and integration tests
├── docs/                # Architecture documentation
└── api/                 # FastAPI routes
```

## Development Methodology

This project follows a **phase-gated development approach**:

1. **Architecture Before Features** — Structural correctness precedes behavioral complexity
2. **Validation-Gated Progression** — Each phase requires verification before continuation
3. **Explicit Constraints** — Progress evaluated against architectural rules, not feature milestones
4. **Engineering Discipline** — Dependency hygiene and modularity prioritized over velocity

## Technical Stack

- **Language:** Python 3.10+
- **Database:** SQLite (development), PostgreSQL (production-ready)
- **Vector Search:** Hash-based embeddings (v1), FAISS/Sentence Transformers (planned)
- **API Framework:** FastAPI
- **Testing:** pytest

## Installation (Development)
```bash
# Clone repository
git clone https://github.com/Talknura/nura.git
cd nura

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start API server
uvicorn api.main:app --reload
```

## Academic Context

This work is being developed as part of:
- **MSIS Capstone Project** — Central Michigan University (Not Yet)
- **Research Focus:** Modular architectures for persistent conversational memory

## Roadmap

- [x] Phase 1: Architectural Boundary Enforcement
- [ ] Phase 2: Cross-Engine Orchestration
- [ ] Phase 3: Dead Code Resolution
- [ ] Phase 4: Interfaces & Abstractions
- [ ] Phase 5: Testing & Validation
- [ ] Phase 6: Scale Preparation

**Proprietary** — All rights reserved. This is a private research project.

## Author

Samuel Sameer Tanguturi  
Master of Science in Information Systems  
Central Michigan University

**Contact:** Tangu1s@cmich.edu
**LinkedIn:** www.linkedin.com/in/tanguturi-sameer-3a5b57303
**Project Started:** October 2025

---

*This project deliberately prioritizes correctness, dependency hygiene, and separation of concerns before any user-facing behavior is activated. The premise is simple: systems that fail to enforce architectural discipline early inevitably accumulate technical debt that limits long-term reasoning quality.*
