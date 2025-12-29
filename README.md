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

**Status:** Validated and complete.

---

### Phase 2: Cross-Engine Orchestration ✅ **COMPLETED**

**Objective:** Activate existing features through coordinated multi-engine workflows.

**Sub-phases:**
- ✅ **2.1 Adaptation Profile Integration** — Connected warmth/formality values to response generation
- ✅ **2.2 Temporal Pattern Detection** — Implemented day-of-week and hour-of-day pattern detection and storage
- ✅ **2.3 Temporal-Aware Retrieval Scoring** — Passed temporal tags from orchestration layer to retrieval
- ✅ **2.4 Memory Summarization** — Implemented automatic summarization when episodic threshold reached

**Outcome:**
- Features activated: 4
- New cross-engine imports: 0
- Boundary violations: 0
- Runtime behavior: Extended (by design)

**Status:** Validated and complete.

---

### Phase 3: Dead Code Resolution ✅ **COMPLETED**

**Objective:** Remove or integrate unused components identified during architecture analysis.

**Sub-phases:**
- ✅ **3.1 Remove Unused Components** — Deleted adaptation_profile.py (unused Pydantic model)
- ✅ **3.2 Fix Unused Parameters** — Removed 'now' parameter from recent(), removed 'desired_type' from score_hit()

**Outcome:**
- Files deleted: 1
- Parameters removed: 2
- Test coverage: Maintained
- Breaking changes: 0

**Status:** Validated and complete.

---

### Phase 4: Interfaces & Abstractions ✅ **COMPLETED**

**Objective:** Add typing.Protocol interfaces for dependency inversion and extract shared utilities.

**Sub-phases:**
- ✅ **4.1 Create Engine Interfaces** — Added Protocol-based interfaces to all 4 engines
- ✅ **4.2 Extract Shared Utilities** — Centralized datetime helpers (dt_to_iso, iso_to_dt)
- ⏸️ **4.3 Event Bus** — Deferred pending architecture clarification

**Outcome:**
- Protocols added: 4 (MemoryEngineProtocol, RetrievalEngineProtocol, TemporalEngineProtocol, AdaptationEngineProtocol)
- New files created: 0 (protocols added to existing engine files)
- Shared utilities: Datetime serialization centralized

**Status:** Validated and complete. Event Bus deferred to future phase.

---

### Phase 5: Testing & Validation ✅ **COMPLETED**

**Objective:** Add comprehensive test coverage for critical paths.

**Sub-phases:**
- ✅ **5.1 Unit Tests** — Created tests for memory classifier, temporal humanizer, adaptation deltas, retrieval ranker
- ✅ **5.2 Integration Tests** — End-to-end chat flow, fact extraction and retrieval, adaptation profile updates
- ✅ **5.3 Evaluation Expansion** — Populated golden_queries.json with test cases

**Outcome:**
- Tests created: 14
- Tests passing: 13
- Tests failing (documented): 1 (keyword matching limitation)
- Pass rate: 92.9%

**Status:** Validated and complete.

---

### Phase 6: Scale Preparation 📋 **PLANNED**

**Objective:** Replace development implementations with production-grade components.

**Scope:**
- Upgrade to sentence-transformers for real embeddings
- Integrate FAISS for vector search
- Database optimizations (indexes, WAL mode, connection pooling)

**Design Principle:** Performance upgrades without architectural changes; engines remain stateless and independently testable.

**Phase 6 initiation contingent upon Phase 5 validation completion.**

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
- [x] Phase 2: Cross-Engine Orchestration
- [x] Phase 3: Dead Code Resolution
- [x] Phase 4: Interfaces & Abstractions
- [x] Phase 5: Testing & Validation
- [ ] Phase 6: Scale Preparation
- [ ] Phase 7: Voice Pipeline Integration
- [ ] Phase 8: Voice Synthesis & Temporal Awareness
- [ ] Phase 9: Logic Injection & Telemetry
- [ ] Phase 10: LLM Integration
- [ ] Phase 11: SSML & Emotive Voice

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
