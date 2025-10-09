<div align="center">
  <img src="https://tellurio-public-assets.s3.us-west-1.amazonaws.com/static/images/afnio-logo-1024x1024.png" width="250">
</div>

# Afnio: Making AI System Optimization Easy for Everyone

<div align="center">
<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#key-concepts">Key Concepts</a> •
  <a href="#contributing-guidelines">Contributing Guidelines</a> •
  <a href="#license">License</a>
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/afnio)](https://pypi.org/project/afnio/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/afnio)](https://pypi.org/project/afnio/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://github.com/Tellurio-AI/afnio/blob/main/LICENSE.md)

</div>

Afnio is a framework for automatic prompt and hyperparameter optimization, particularly designed for complex AI systems where Language Models (LMs) are employed multiple times in workflows, such as in LM pipelines and agent-driven architectures. Effortlessly build and optimize AI systems for classification, information retrieval, question-answering, etc.

## Quickstart

Get started with Afnio in six steps, or try it instantly in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tellurio-AI/tutorials/blob/main/facility_support/facility_support_sentiment.ipynb)

1. Install the Afnio SDK with [pip](https://pip.pypa.io/en/stable/):

```bash
pip install afnio
```

2. Set the API key for the LLM model you want to use as an environment variable (OpenAI for this quickstart). Get your key from [OpenAI dashboard](https://platform.openai.com/api-keys).

```bash
export OPENAI_API_KEY="your-api-key"
```

3. Log in to [Tellurio Studio](https://platform.tellurio.ai/) and paste your API key when prompted. Create or view your API keys under the [API Keys](https://platform.tellurio.ai/settings/api-keys) page.

```bash
afnio login
```

4. Copy and run this sample code to optimize your AI agent and track its quality metrics. Your first Run will appear in [Tellurio Studio](https://platform.tellurio.ai/). Your system's checkpoints will be saved under the local `checkpoint/` directory created in the same path where you executed the script.

   _This example uses [Meta's Facility Support Analyzer dataset](https://github.com/meta-llama/prompt-ops/tree/main/use-cases/facility-support-analyzer) to classify enterprise support emails as positive, neutral, or negative. **Expect accuracy to improve from 66.4% ±1.5% to 80.8% ±12.5% — a +14.5% absolute gain.**_

````python
import json
import re

import afnio
import afnio.cognitive as cog
import afnio.cognitive.functional as F
import afnio.tellurio as te
from afnio.models.openai import AsyncOpenAI
from afnio.trainer import Trainer
from afnio.utils.data import DataLoader, WeightedRandomSampler
from afnio.utils.datasets import FacilitySupport

# Initialize Project and experiment Run
run = te.init("your-username", "Facility Support")


# Compute per-sample weights to balance the training set
def compute_sample_weights(data):
    with te.suppress_variable_notifications():
        labels = [y.data for _, (_, y, _) in data]
        counts = {label: labels.count(label) for label in set(labels)}
        total = len(data)
    return [total / counts[label] for label in labels]


# Prepare data and loaders
train_data = FacilitySupport(split="train", root="data")
test_data = FacilitySupport(split="test", root="data")
val_data = FacilitySupport(split="val", root="data")

weights = compute_sample_weights(train_data)
sampler = WeightedRandomSampler(weights, num_samples=len(train_data), replacement=True)

BATCH_SIZE = 33
train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, seed=42)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, seed=42)

# Define prompt and response format
sentiment_task = "Read the provided message and determine the sentiment."
sentiment_user = "Read the provided message and determine the sentiment.\n\n**Message:**\n\n{message}\n\n"
SENTIMENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": "sentiment_response_schema",
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"],
                },
            },
            "additionalProperties": False,
            "required": ["sentiment"],
        },
    },
}

# Set up LM model clients used for forward, backward passes and optimization step
afnio.set_backward_model_client("openai/gpt-5", completion_args={"temperature": 1.0, "max_completion_tokens": 32000, "reasoning_effort": "low"})
fw_model_client = AsyncOpenAI()
optim_model_client = AsyncOpenAI()


# Define the sentiment classification agent
class FacilitySupportAnalyzer(cog.Module):
    def __init__(self):
        super().__init__()
        self.sentiment_task = cog.Parameter(data=sentiment_task, role="system prompt for sentiment classification", requires_grad=True)
        self.sentiment_user = afnio.Variable(data=sentiment_user, role="input template to sentiment classifier")
        self.sentiment_classifier = cog.ChatCompletion()

    def forward(self, fwd_model, inputs, **completion_args):
        sentiment_messages = [
            {"role": "system", "content": [self.sentiment_task]},
            {"role": "user", "content": [self.sentiment_user]},
        ]
        return self.sentiment_classifier(fwd_model, sentiment_messages, inputs=inputs, response_format=SENTIMENT_RESPONSE_FORMAT, **completion_args)

    def training_step(self, batch, batch_idx):
        X, y = batch
        _, gold_sentiment, _ = y
        pred_sentiment = self(fw_model_client, inputs={"message": X}, model="gpt-4.1-nano", temperature=0.0)
        pred_sentiment.data = [json.loads(re.sub(r"^```json\n|\n```$", "", item))["sentiment"].lower() for item in pred_sentiment.data]
        loss = F.exact_match_evaluator(pred_sentiment, gold_sentiment)
        return {"loss": loss, "accuracy": loss[0].data / len(gold_sentiment.data)}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        constraints = [
            afnio.Variable(
                data="The improved variable must never include or reference the characters `{` or `}`. Do not output them, mention them, or describe them in any way.",
                role="optimizer constraint",
            )
        ]
        optimizer = afnio.optim.TGD(self.parameters(), model_client=optim_model_client, constraints=constraints, momentum=3, model="gpt-5", temperature=1.0, max_completion_tokens=32000, reasoning_effort="low")
        return optimizer


# Instantiate agent and trainer
agent = FacilitySupportAnalyzer()
trainer = Trainer(max_epochs=5)

# Evaluate the agent on the test set before training (baseline performance)
llm_clients = [fw_model_client, afnio.get_backward_model_client(), optim_model_client]
trainer.test(agent=agent, test_dataloader=test_dataloader, llm_clients=llm_clients)

# Train the agent on the training set and validate on the validation set
trainer.fit(agent=agent, train_dataloader=train_dataloader, val_dataloader=val_dataloader, llm_clients=llm_clients)

run.finish()
````

5. View live metrics, compare Runs, and share results with your team.

<div align="center">
  <img src="https://tellurio-public-assets.s3.us-west-1.amazonaws.com/static/images/tellurio-studio-quickstart-plots.png" width="90%">
</div>

6. Run your optimized AI agent on the test set to see how it performs, or on new data! Check out how on our Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tellurio-AI/tutorials/blob/main/facility_support/facility_support_sentiment.ipynb)

## Key Concepts

- **Accelerated AI System Development:** Ship complex AI systems faster thanks to high-level UX and easy-to-debug runtime.
- **State-of-the-Art Performance:** Leverage built-in optimizers to automatically refine prompts and tune model parameters for any LM task, ensuring optimal performance.
- **LM Agnostic:** Decouple prompts and parameters from application logic, reducing LM model selection to a single hyperparameter in Afnio’s optimizers. Seamlessly switch between models without any additional rework.
- **Minimal and Flexible:** Pure Python with no API calls or dependencies, ensuring seamless integration with any tools or libraries.
- **Progressive Disclosure of Complexity:** Leverage diverse UX workflows, from high-level abstractions to fine-grained control, designed to suit various user profiles. Start simple and customize as needed, without ever feeling like you’re falling off a complexity cliff.
- **_Define-by-Run_ Scheme:** Your compound AI system is dynamically defined at runtime through forward computation, allowing for seamless handling of complex control flows like conditionals and loops, common in agent-based AI applications. With no need for precompilation, Afnio adapts on the fly to your evolving system.

## Contributing Guidelines

💻 Would love to contribute? Please follows our [contribution guidelines](CONTRIBUTING.md).

## License

Afnio is open-source under the GNU Affero General Public License v3 (AGPLv3).

You can freely use Afnio — in personal, research, or commercial projects.  
You don’t need to open-source your code; the license only applies if you modify Afnio itself and share that version publicly.

We keep Afnio open so everyone can build freely while helping the project grow.

💌 Questions or ideas? We’d love to hear from you at contact@tellurio.ai.
