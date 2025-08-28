1) Run the SFT classifier with 512, 1024, 2048, 4096 and full 
    a) record their accuracy, think proportion, answer proportion and length
2) Run the GRPO with 512, 1024, 2048, 4096 and full
    a) record their accuracy, think proportion, answer proportion and length
3) Run the GRPO with shorten answer reward with 512, 1024, 2048, 4096 and full
    a) record their accuracy, think proportion, answer proportion and length


Actually, do this whole cycle with each weight. We should be getting the sample n from an env variable

One pretty cool graph would be to have the accuracy and training samples as the axis. If the training length changes as we train for longer, we could have that too.

Questions?
    - How many mor epochs (if any) do we need to match the SFT performance with the GRPO?
    - What happens if we decrease the number of generations per sample?


Other stuff:

1) Plot model trajectories onto 2d map
2) Compare the trajectories between different models
3) Use the logprobs as a models embedding to compare similarity.
4) Plot a cloud of the embeddings of the generations with different colors by model


########################

● Experimental Objective and Design Decisions

  The primary objective of this experiment is to conduct a fair and systematic comparison between Supervised Fine-Tuning (SFT)
  and Group Relative Policy Optimization (GRPO) for mathematical reasoning tasks. We aim to evaluate whether GRPO's reinforcement
   learning approach provides meaningful improvements over standard supervised learning when both methods have access to
  identical training data.

  Training Methodology

  To ensure experimental fairness, we designed a controlled comparison where both training methods operate on exactly the same
  dataset samples. For each experiment with n_samples training examples, we train two models: (1) a pure SFT model that trains
  directly on all n_samples using supervised learning, and (2) a GRPO model that first undergoes SFT pre-training on a subset of
  samples, then applies GRPO training to the full dataset. This design reflects the realistic usage pattern of GRPO, which
  typically requires an SFT initialization, while ensuring both methods have equivalent data access.

  Key Design Decisions

  Data Parity: Both training approaches see identical training samples, eliminating data access as a confounding variable. Any
  performance differences can be attributed to the training methodology rather than dataset variations.

  Methodological Authenticity: The GRPO training follows standard practices by initializing from an SFT checkpoint, using a
  smaller subset for initial supervised training before applying the full GRPO procedure. This mirrors real-world GRPO
  implementations while maintaining fairness.

  Scalability Testing: Rather than testing a single sample size, we conduct experiments across multiple values of n_samples,
  always ensuring matching dataset sizes between SFT and GRPO. This approach reveals how the relative effectiveness of each
  method scales with data availability.

  Hyperparameter Considerations: While we maintain identical core parameters where possible (model architecture, epochs, weight
  decay, random seeds), some hyperparameters necessarily differ between methods due to their distinct optimization dynamics. GRPO
   uses a significantly lower learning rate and different batch sizing, reflecting the requirements of policy optimization versus
   direct supervised learning.

  This experimental design provides a robust framework for evaluating whether GRPO's additional complexity and computational
  overhead translates to meaningful performance improvements over traditional supervised fine-tuning in mathematical reasoning
  tasks.