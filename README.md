# ⚙️ EvalCore

<p align="center">
  <strong>Measure the intelligence. Improve the outcome.</strong>
</p>

<p align="center">
  An evaluation-first toolkit for building reliable LLM applications.
</p>

<p align="center">
  <a href="https://github.com/ATLURI0001/EvalCore">GitHub</a> ·
  <a href="https://docs.ragas.io/">Documentation</a> ·
  <a href="https://discord.gg/5djav8GGNZ">Community</a>
</p>

---

## What is EvalCore?

**EvalCore** is a developer-focused evaluation toolkit for LLM applications.

It helps you move beyond *“the model seems to work”* toward measurable answers:

* Is the response accurate?
* Is the retrieved context relevant?
* Does the answer stay grounded in the available information?
* How does a new model or prompt compare with the previous one?
* Where does the application need improvement?

Instead of treating evaluation as a final inspection, EvalCore makes it part of the engineering workflow.

> **Build the application. Measure its behavior. Improve what matters.**

## Why evaluation matters

LLM applications can produce convincing answers without consistently producing correct ones.

A useful evaluation workflow turns that uncertainty into evidence:

```text
LLM Application
       │
       ▼
   Test Dataset
       │
       ▼
    Evaluation
       │
       ▼
  Quality Signals
       │
       ▼
  Engineering Decisions
       │
       └──────────────► Continuous Improvement
```

EvalCore is built around this feedback loop.

## Core capabilities

| Capability                    | Purpose                                                                     |
| ----------------------------- | --------------------------------------------------------------------------- |
| 🎯 **Objective metrics**      | Evaluate application outputs using measurable quality signals.              |
| 🧪 **Test generation**        | Create evaluation datasets for different scenarios and use cases.           |
| 🔍 **Quality analysis**       | Identify weaknesses in retrieval, generation, and overall responses.        |
| 🔗 **Framework integrations** | Connect evaluation workflows with the tools used to build LLM applications. |
| 📊 **Feedback loops**         | Use evaluation results to guide continuous improvement.                     |

## Installation

```bash
pip install evalcore
```

> **Note:** Replace the package name above with the actual published package name when you release the project.

## Quick start

The following example demonstrates the evaluation-first workflow:

```python
import asyncio

from openai import AsyncOpenAI
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory


# Configure the evaluation model
client = AsyncOpenAI()
llm = llm_factory("gpt-4o", client=client)


# Define what you want to measure
metric = DiscreteMetric(
    name="summary_accuracy",
    allowed_values=["accurate", "inaccurate"],
    prompt="""Evaluate if the summary is accurate and captures key information.

Response: {response}

Answer with only 'accurate' or 'inaccurate'."""
)


# Evaluate an application output
async def main():
    score = await metric.ascore(
        llm=llm,
        response="The summary of the text is..."
    )

    print(f"Score: {score.value}")
    print(f"Reason: {score.reason}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Evaluation workflow

A typical EvalCore project follows a simple progression:

### 1. Define the quality signal

Decide what matters for your application:

* Accuracy
* Relevance
* Faithfulness
* Completeness
* Response quality
* Custom domain-specific criteria

### 2. Prepare evaluation data

Use representative examples from your application, including the cases where quality matters most.

### 3. Run evaluations

Apply metrics to your application outputs and collect structured results.

### 4. Inspect the results

Look for patterns, failure cases, and changes in performance.

### 5. Improve and repeat

Update your prompts, retrieval strategy, models, or application logic, then evaluate again.

**The goal is not just a score. The goal is a better system.**

## Project structure

```text
EvalCore/
├── docs/                 # Documentation
├── examples/             # Evaluation examples
├── tests/                # Test suite
├── src/                  # Core implementation
├── pyproject.toml        # Project configuration
└── README.md
```

## Roadmap

* [ ] Core evaluation metrics
* [ ] RAG evaluation workflows
* [ ] Test-data generation
* [ ] Custom evaluation criteria
* [ ] Experiment comparison
* [ ] Production feedback workflows
* [ ] Agent evaluation support
* [ ] LLM benchmarking
* [ ] Prompt evaluation
* [ ] Workflow evaluation

## Contributing

Contributions are welcome.

```bash
git clone https://github.com/ATLURI0001/EvalCore.git
cd EvalCore
```

Create a feature branch, make your changes, and open a pull request.

```bash
git checkout -b feature/your-feature
git commit -m "your meaningful commit message"
git push origin feature/your-feature
```

## License

This project is licensed under the **Apache License 2.0**.

---

<p align="center">
  <strong>⚙️ Build with confidence. Evaluate with evidence.</strong>
</p>
