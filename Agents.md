# 📘 AEON-CODEX – AGENTS.md

schema_version: "1.1"
description: >
  Manifest lebender Agenten im AEON-Codex. Jeder Agent ist eine Aktivierungszelle
  innerhalb eines symbolischen SelfAudit-Systems. Diese Version enthält Visualisierung,
  Rechteverwaltung und Dokumentationsverweise.

visual: "agents_chain.svg"
test_mode: true
default_role: "dev"

roles:
  - id: "admin"
    canAccess: ["*"]
  - id: "dev"
    canAccess: ["CodexAuditAgent", "EvolverGPT", "FragmentMapper"]
  - id: "guest"
    canAccess: []

---

## 🧠 Agent: CodexAuditAgent
start: "mandala-sync.ts"
modules:
  - "audit-core.ts"
  - "depthvalue-core.ts"
  - "crepJudgeGPT"
activate_if:
  depth.lnSum: "> 14"
  crep.state: "emergence"
roles_allowed: ["admin", "dev"]
docs: "docs/agents/codexaudit.md"

## 🧬 Agent: EvolverGPT
start: "codexwork.yaml"
modules:
  - "codex-evolver.ts"
  - "crepdecision-core.ts"
activate_if:
  crep.score: ">= 0.6"
  depth.symbolics.contains: "🌪"
roles_allowed: ["admin", "dev"]
docs: "docs/agents/evolvergpt.md"

## 🔍 Agent: FragmentMapper
start: "fragmented_conversation.json"
output: "codexwork.yaml"
roles_allowed: ["admin", "dev"]
docs: "docs/agents/fragmentmapper.md"

## 🔁 Agent: SyncRunner
start: "codexsync.yaml"
roles_allowed: ["admin"]
docs: "docs/agents/syncrunner.md"

## 🔐 Agent: PactDepthGatekeeper
start: "pact-depth-rules.ts"
activate_if:
  depth.lnSum: "> 16"
docs: "docs/agents/pactdepthgatekeeper.md"

## 📦 Agent: DepthBundleExporter
output:
  - "sigillin_depth_bundle.sigil.json"
  - "depth_index.md"
  - "irrational_matrix.wav"
  - "mandala_depth_*.svg"
trigger:
  - "manual"
  - crep.event: "bundleReady"
docs: "docs/agents/depthbundleexporter.md"
