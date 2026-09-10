# Secure Cloud for Agent-Generated Small Software

DATA 298A — Team 5 — Topic 7

## Project Overview

This project explores a secure multi-tenant cloud platform for deploying and sharing AI agent-generated micro-applications.

The platform focuses on four main capabilities:

1. Code and dependency risk classification
2. Least-privilege permission inference
3. Runtime anomaly detection
4. Resource-aware scheduling and cold-start optimization

The goal is to allow nontechnical users to deploy small AI-generated applications while reducing security risks related to vulnerable code, excessive permissions, malicious runtime behavior, and inefficient resource usage.

## Proposed Models

### Model 1 — Code and Dependency Risk Classifier
Uses code representations, dependency information, and security evidence to identify risky or vulnerable applications.

### Model 2 — Permission Inference
Infers the minimum permissions required by an application using static code information and runtime behavior.

### Model 3 — Runtime Anomaly Detection
Uses system-call sequences to identify abnormal or potentially malicious runtime behavior.

### Model 4 — Resource Scheduling
Predicts application demand and supports pre-warming and resource allocation decisions.

## Published Baseline

Our published baseline is:

**CodeBERT evaluated on the PrimeVul benchmark**

- Paper: *Vulnerability Detection with Code Language Models: How Far Are We?*
- Published at ICSE 2025
- Result to reproduce: **Table V**
- Target: **CodeBERT F1 = 20.86% on PrimeVul**

Baseline reproduction has not yet been completed.

## Data Sources

Current planned data sources include:

- PrimeVul — baseline reproduction
- SecBench.js / CVEfixes / OSV — Model 1
- AuthBench and team-created labels — Model 2
- ADFA-LD — Model 3
- Azure Functions traces — Model 4
- MicroAppSec-100 — team-created end-to-end benchmark

Some dataset access, licensing, and preprocessing steps are still pending.

## Evaluation

Baseline reproduction will primarily use F1 score.

For the integrated platform, the proposed target is:

- Exploit-blocking recall ≥ 90%
- False-denial rate ≤ 5%

Evaluation will later use the team-created MicroAppSec-100 benchmark.

## Current Status

Checkpoint 1 work includes:

- Literature review and state-of-the-art analysis
- Dataset and benchmark investigation
- Published baseline selection
- Four-model methodology design
- Evaluation criteria definition
- Risk and compute planning

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── baseline/
├── docs/
├── src/
└── tests/
```
