# Future Work and Ideas

### MCQDataset
- Should work with a local dataset (.txt, .csv, .jsonl for instance as well)
- Should wrap the dataset download function from HF into a utility

### Rate Limit
- Google provides quite high rate limits in that we have about 1500 RPD and 15 RPM.
- We should implement a rate limiter to avoid hitting the limits.

### Generate sample
- It might be useful to generate a sample of the dataset to inspect it before the full generation process.
- Think of it like a `dataset.generate_sample(providers, num_samples=10)`.

### Modularize and Break Down some Dataset Classes
- Some dataset classes like UltraChatDataset are too large and complex.
- We should break them down into smaller, modular pieces.
- I prefer to wait for having more datasets before doing this so that we can have a better overview of the needs.

### Capability to interrupt and resume the generation process
- It might actually be desired.

### Async/Batch generation
- For a few hundred samples it is OK to wait a bit. However, generating thousands of samples should be done Async/Batch as it becomes unacceptable for a user to wait so long.
- Check what options LiteLLM offers. This is another reason to migrate to LiteLLM.

### Reasoning Dataset
- Simple DeepSeek distillation?
- Agentic Generation?
- We can have a look at CamelAI, but also check how other reasoning dataset are generated.
- First I would like to switch to using LiteLM for inference, and also test some approaches. Then we work on it.

### Tests
- Improve the test suite

### EvolInstruct not implemented
- At the moment EvolInstruct is not implemented in PreferenceDataset Generation.
- While this is not critical at this stage, it would be great to have it soon.


### Import HF dataset sample
- When reusing a dataset from the HF Hub, check if we can just download a random sample of like 100 examples.
- This would be much faster and more efficient rather than the entire dataset which sometimes is huge.

### Pushing dataset to Argilla for Human Review / Feedback
- Having human feedback on generated data is important for quality control.
- It's quite straightfoward to push a dataset to argilla to collect human feedbacks