# Data Sources

This document summarizes the current data sources planned for the four project models and the published baseline.

## Published Baseline

**PrimeVul**
- Purpose: baseline reproduction for CodeBERT
- Published result: CodeBERT on PrimeVul, Table V
- Target F1: 20.86%
- Access status: public data and code identified
- Reproduction status: not yet completed

## Model 1 — Code and Dependency Risk Classifier

**Planned sources**
- SecBench.js
- CVEfixes
- OSV

**Role**
Provide code vulnerability, dependency, and security-risk information for training and evaluation.

**Current status**
- SecBench.js usage/license clarification is still pending
- OSV is publicly available
- Additional preprocessing and provenance checks are still needed

## Model 2 — Permission Inference

**Planned sources**
- AuthBench
- Team-created permission labels

**Role**
Support least-privilege permission inference and validation.

**Current status**
- Project-specific training labels still need to be created
- AuthBench will be used as a reference rather than a complete training-ready dataset

## Model 3 — Runtime Anomaly Detection

**Planned source**
- ADFA-LD

**Role**
Provide labeled system-call traces for runtime anomaly detection.

**Current status**
- Academic-use terms identified
- Official download access is currently unresolved
- Alternative sources may be considered if access remains blocked

## Model 4 — Resource Scheduling

**Planned sources**
- Azure Functions 2019 trace
- Azure Functions 2021 trace as a possible supplement

**Role**
Provide invocation and workload traces for demand prediction, pre-warming, and scheduling.

**Current status**
- Azure Functions 2019 data is available
- Azure Functions 2021 extraction is still pending

## End-to-End Benchmark

**MicroAppSec-100**

MicroAppSec-100 is not an existing public dataset. It is a team-created benchmark that we plan to construct for integrated platform evaluation.

Proposed design:
- 20 application families
- 5 variants per family
- 100 applications total

The benchmark will be used to evaluate exploit-blocking recall, false-denial rate, deployment success, and overall platform behavior.

## Overall Data Readiness

Dataset readiness is currently **Partial / Pending**.

Some public sources are already available, while licensing, download access, preprocessing, and team-created labels still need to be completed.
