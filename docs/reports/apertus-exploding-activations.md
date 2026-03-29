# Active Interpretability with Hallucination Probes: Exploding Activations in Apertus-8B-Instruct-2509

**Author:** Tymoteusz Kwieciński
**Date:** 2026-03-17

---

## TL;DR

This post covers the investigation process why hallucination probes trained on Apertus-8B-Instruct-2509 were significantly less stable than probes trained on Llama-3.1-Instruct. The main sympthom was a really strong degradation of probe quality in deeper layers and much noisier training loss. We hypothesize that the worse performance comes from exploding activations in deeper layers of Apertus-8B-Instruct-2509 model and proposed few fixes, improving the overall probe performance from `0.7025` to 	`0.8961` AUC and `0.3837` to `0.6802` recall at `0.1` false positive ratio.


This post covers our investigation into why hallucination probes trained on
Apertus-8B-Instruct-2509 were significantly less stable than probes trained on
Llama-3.1-8B-Instruct. The main symptom was a strong degradation of probe
quality in deeper layers and much noisier training loss. We hypothesize that
this comes from **exploding activations** in deeper layers of the Apertus model, which is a problem where hidden state magnitudes grow far beyond what the probe's
optimizer can handle reliably. By addressing this directly, we improved the
overall probe performance from `0.7025` to `0.8961` AUC and from `0.3837` to
`0.6802` recall at `0.1` false positive ratio.

---

## Problem description 

### Background

We extend the work of [Obeso et al.](https://arxiv.org/abs/2509.03531) by
reproducing their hallucination probe methodology for Apertus-8B-Instruct-2509
and exploring ways to improve probe performance on this model.





Probes trained in the paper are used to detect hallucinated entities in the long-form text. This task is meant to reasemble the usual conversation with the model. A probe in this context is a small model trained on top of an LLM activations from a given layer, which performs a per-token classification task. This approach was proven very useful, e.g. in [Zou et. al.](https://arxiv.org/pdf/2310.01405).  In our setting, the probe was predicting whether a given token from the text is a hallucination or not. 

The probes in that paper are designed to detect hallucinated entities in
long-form text — the kind of open-ended, multi-sentence responses you'd get in a natural conversation with a chatbot. The idea is that an LLM's internal activations (the hidden states produced at each transformer layer) carry rich information about what the model *knows* and *doesn't know.* A **probe** is a small classifier trained on top of those activations to predict, for each token, whether it's part of a hallucinated span or not. This approach of reading off model internals has been
shown to work well across a variety of tasks — for instance, [Zou et al.
(2023)](https://arxiv.org/pdf/2310.01405) demonstrated that linear probes on
residual stream activations can reliably extract concepts like truthfulness and sentiment, which motivates using them here, this time in more natural setting.



### The dataset description


An example from such dataset, that is meant to reasemble natural conversation with an LLM is shown below. One datapoint in the dataset consists of a user question followed by a model response. Certain spans are annotated as supported or not supported by web search:

---

**User**: Tell me about Formula 1's current testing regulations, particularly focusing on restrictions during the season, pre-season testing allowances, rookie driver testing requirements, tire testing protocols, and any special testing permissions for development work.

**Model**: As of my knowledge cutoff in 2024, Formula 1's testing regulations have undergone significant changes over the years, with the primary goal of reducing costs and promoting more competitive racing. Here's an overview of the current testing regulations, focusing on the points you've mentioned:

### 1. Restrictions during the season
- **Seasonal Testing**: *Formula 1 teams are not allowed to conduct any official testing during the season*. This is to ensure that all teams have an equal opportunity to prepare for the championship and to prevent any team from gaining an unfair advantage through extensive testing.
- **Private Testing**: Teams are allowed to conduct private testing, but this is heavily restricted. Private testing is typically limited to a few days per year and is usually conducted on private tracks or simulators.

### 2. Pre-season testing allowances
- **Pre-season Testing**: Formula 1 teams are allowed to conduct pre-season testing, but this is also heavily restricted. The pre-season testing is usually conducted at the beginning of the year, typically at the same track where the first race of the season will be held.
- **Testing Days**: The number of testing days is limited. For example, in 2024, each team was allowed a maximum of *100 kilometers of testing per day*, with a total of *1500 kilometers allowed across all pre-season testing sessions*.




Below you can see the annoated spans - fragments of text that were confirmed to be false or true by a web-search-based annotation pipeline:

```

[
  {
    "index": 356,
    "label": "Supported",
    "span": "Formula 1 teams are not allowed to conduct any official testing during the season",
    "verification_note": "The FIA regulations restrict in-season testing to control costs and maintain fairness."
  },
  {
    "index": 875,
    "label": "Supported",
    "span": "Pre-season Testing",
    "verification_note": "Pre-season testing is allowed under FIA regulations with specified limits."
  },
  {
    "index": 1260,
    "label": "Not Supported",
    "span": "100 kilometers of testing per day",
    "verification_note": "No specific information found on a 100 kilometers per day limit; regulations normally specify total pre-season test days instead."
  },
  {
    "index": 1311,
    "label": "Not Supported",
    "span": "1500 kilometers allowed across all pre-season testing sessions",
    "verification_note": "No specific mileage limit per season found; FIA usually regulates the number of test days, not mileage."
  }
]
```

---


The dataset [`LongFact++`](https://huggingface.co/datasets/obalcells/longfact-annotations), which was used in the paper to train hallucination probes was collected in 3 steps:
1. Question prompt collection
2. Model's answers generation
3. Generation annotation using larger LLM with websearch tool

Using a web-search-based annotation is important because it grounds the labels in real sources. An LLM judge without internet access might confidently confirm a false claim if it appears plausible, whereas a grounded web search can actually check it. Obeso et al. validated the pipeline in two ways. First, a human annotator independently labeled a random sample of 50 entity spans and agreed with the LLM's labels in 84% of cases — though this only measures precision, not recall. Second, on a controlled dataset built by paraphrasing Wikipedia passages into conversational text and injecting known factual errors, the annotation pipeline detected 80.6% of injected hallucinations (729/904), with a false positive rate of 15.8% on unchanged factual content.


The questions were sourced from [`LongFact`](https://arxiv.org/abs/2403.18802),
a Google-curated corpus of general-knowledge questions. Obeso et al. extended
this to ~20k questions across diverse topics, then generated answers using
various open-source models (Llama-8B-Instruct, Llama-70B-Instruct,
Mistral-Small-24B-Instruct-2501, and others), and finally annotated them using
Anthropic Claude Sonnet 4 with a web search tool.

To reproduce probes for Apertus, we had to run both the generation and
annotation pipelines ourselves, which produced our version of `LongFact++`,
publicly available on
[HuggingFace](https://huggingface.co/datasets/tkwiecinski/longfact-test-split).
We generated data for both Apertus-8B-Instruct-2509 and Llama-3.1-8B-Instruct.
The full details of this reproduction are in the [project
report](https://github.com/swiss-ai/feature-probes/blob/main/docs/reports/hallucination_detection.pdf).
The original pipeline documentation was not well enough documented and shared, thus a perfect 1:1 match
wasn't possible, but the resulting datasets are qualitatively and
quantitatively close to the originals, as outlined in the section below.


### Probe training process

Each probe is trained on the activations produced after a specific transformer layer. We followed the original paper and used activations from after layer 30 (both Apertus-8B-Instruct-2509 and Llama-3.1-8B-Instruct have 32 layers in total). Neither the original authors nor we initially performed a systematic layer-wise comparison. this turned out to be a significant omission, as discussed below.

Probes were trained with a simple cross-entropy loss, commonly used in classification tasks. 


Probes are trained with a standard cross-entropy loss, which is the typical
choice for binary classification tasks. To further improve performance, the
original paper also proposed using **LoRA** (Low-Rank Adaptation,
[Hu et al., 2021](https://arxiv.org/abs/2106.09685)). LoRA are small trainable adapters that project the activations into a lower-dimensional space before the classifier head. The intuition is that LoRA can surface task-relevant structure in the activations that a linear probe operating on the full hidden dimension might miss. For more details on the full training setup, refer to the [reproduced paper](https://arxiv.org/abs/2509.03531).


## The problem

While working on the project, we noticed that the probe trained on Apertus-8B-Instruct-2509 activations, performs much worse on the training set generated with Apertus-8B-Instruct-2509 than the probe trained on Llama-8B-Instruct activations. To better understand this behaviour we performed few evaluations.  <<citation here>>


While working on the project, we noticed that the probe trained on
Apertus-8B-Instruct-2509 activations performs much worse than the equivalent
probe trained on Llama-3.1-8B-Instruct activations. This result was particularly suprising, because we evaluated both probes on Apertus-generated completions. To understand this better, we ran a layer-wise evaluation: training a separate probe on activations from a subset of the 32 layers, then measuring AUC on a held-out test set.


![layers](./media/layers.png)


As you can see in the plot above, the Apertus-8B-Instruct-2509 probe performance drops significantly, after 16th layer. This is unexpected, because the deeper layers in transformers tend to express more abstract concepts. < cite here > Final training loss also explodes around that layer. What is even more suprising, is that the probe trained on activations from Llama-8B-Instruct (a completely different model) performs much better than the probe trained on the Apertus-8B-Instruct-2509 activations.


## Diagnosing the Instability

### Loss scale comparison

The first thing we checked was the raw scale of the training loss. The
cross-entropy loss for the Apertus probe is roughly **100× larger** than for
the Llama probe, and it's much more spiky across training steps:

![loss](./media/loss.png)

The plot above shows the scale. Even though the loss is much larger in scale, the training converges. 


### Dataset validity

To verify if the source of the observed issue comes from the model, and not the data that was generated by our pipeline, we revisited the dataset validity. We compared both
datasets along several dimensions: annotation quality, hallucination rates,
completion lengths, span lengths, and train/test overlap. The table below shows
that our datasets are very similar to those from Obeso et al., both in terms of
volume and distribution:



| Source | Model | Split | Rows | Spans/Row | Invalid Rate | Halluc. Rate | Completion Length (mean) | Completion Length (median) | Span Length (mean) | Span Length (90th percentile) |
|:-------|:--------------|:------|-------:|----------:|-------------:|-------------:|------------------:|-----------------:|----------------:|---------------:|
| ours | Apertus-8B | train | 17,986 | 9.59 | 0.025 | 0.236 | 3,014 | 2,778 | 25.66 | 52 |
| ours | Apertus-8B | test | 1,993 | 9.61 | 0.025 | 0.240 | 2,987 | 2,733 | 26.08 | 54 |
| ours | Llama-3.1-8B | train | 17,959 | 11.99 | 0.028 | 0.264 | 3,918 | 3,728 | 25.31 | 50 |
| ours | Llama-3.1-8B | test | 1,996 | 11.78 | 0.029 | 0.269 | 3,915 | 3,708 | 25.71 | 50 |
| paper | Llama-3.3-70B | train | 7,959 | 15.59 | 0.021 | 0.261 | 3,890 | 3,735 | 20.72 | 41 |
| paper | Llama-3.1-8B | train | 7,919 | 14.44 | 0.023 | 0.367 | 3,687 | 3,474 | 22.43 | 45 |
| paper | Mistral-24B | train | 1,534 | 14.20 | 0.034 | 0.170 | 3,567 | 3,560 | 19.87 | 39 |
| paper | Qwen2.5-7B | train | 1,524 | 12.64 | 0.045 | 0.350 | 3,782 | 3,757 | 21.60 | 43 |
| paper | Gemma-2-9B | train | 1,495 | 10.94 | 0.048 | 0.186 | 2,837 | 2,781 | 19.47 | 37 |


The hallucination rates (~24% for Apertus, ~26% for Llama) are well-matched,
and there are no train/test leaks. Apertus completions are slightly shorter on average, which could marginally affect span density, but nothing here
explains a that significant performance drop for the Apertus probe. The data is definetely not the issue.



### Activations clustering


With the dataset ruled out, we looked directly at the activations. One useful sanity check is to visualize them with PCA. If the activations contain signal that is relevant for hallucination detection, you would expect some separation between hallucinated and supported tokens. 


![layers](./media/activations.png)

Both Apertus and Llama activations show some clustering, even though the characteristic of the clusters differ a lot. 

The key difference between two models' activations is the **spread**: Apertus activations are far more dispersed in the PCA projection (take a look at the scale of the plot). It looks like the signal for hallucination detection is there, but it is much more noisy.


### Activations norm

To quantify what we saw in the PCA, we directly measured the mean L2 norm of
activations at each layer for both models:

![layers](./media/explosion.png)

While growing activation norm accross layers is expected in transformer models, since the residual stream accumulates contributions at every layer (see [this post](https://turntrout.com/residual-stream-norms-grow-exponentially-over-the-forward-pass)).

What's not expected is the scale we see here: Apertus activations are over
**100× larger in magnitude** than Llama's in the deeper layers, which
quantitatively matches the 100× loss difference we observed earlier. Such large activations might pose a problem for further calculations with them, particularly because usual computational methods are adapted to work with numbers of small magnitude.


## Hypothesis


The evidence consistently supports one explanation: **exploding activations**
in Apertus-8B-Instruct-2509. The activations in deeper layers reach magnitudes
that cause numerical instability in `bf16` arithmetic, scatter gradients during
optimization, and ultimately prevent the probe from learning a reliable
decision boundary.

This issue was independently raised by Julian Minder, whom we contacted for
additional context. He also pointed us toward several potential fixes. 

One long-term fix would be to address the root cause in the model itself and
release an Apertus version without this exponential growth. In the meantime,
however, we wanted to verify the hypothesis empirically: if the instability
truly comes from activation magnitudes, then interventions that normalize or
dampen those magnitudes should noticeably improve probe stability and
performance.



## Proposed solutions



We tried four interventions, each targeting the problem from a slightly
different angle — numerical precision, optimization dynamics, input
normalization, and learned projection:

- **Stronger precision (`fp32` instead of `bf16`)**: `bf16` has limited
  range for large numbers; `fp32` can represent the same values with
  significantly better accuracy, which might prevent rounding errors from
  cascading during training.
- **Smaller learning rate**: when activations are
  large, gradients can be large too, and a big learning rate amplifies this
  into unstable updates. Slowing down the optimizer gives it more room to
  navigate around the instability.
- **LayerNorm before the probe**: normalizing the activations to zero mean
  and unit variance before feeding them to the probe directly removes the
  magnitude problem at the input level. This is essentially the same trick
  used inside transformers themselves to keep training stable
  ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)).
- **LoRA adapters**:  as described in the original paper, LoRA learns a
  low-rank projection of the activations. Beyond improving task alignment,
  this projection may act as an implicit compression step that filters out
  some of the high-variance noise in the Apertus activations.



## Results

It turned out that the proposed solutions helped with stabilizing the runs and improving performance, though to very different degrees. Now, we will go over each of the solutions.

### Stronger precision

Using `fp32` instead of `bf16` for probe training improved stability. The
loss became less noisy across seeds. This improvement though had no meaningful impact on final performance:

![precision](./media/precision.png)

The intuition behind using a stronger precision to train the probe is that the float with more bits can express the large number with much better precision. Since observed activations were really large, we suspected that by changing the precision to `float32` the probe might be able to produce more stable results. It doesn't not change the geometry of the activations. The probe is still trying to find a decision boundary in a space with very high variance.


### Smaller learning rate



Reducing the learning rate only slightly, from 1e-3 to 3e-4 also helped to stabilize the runs and slightly improved the performance:

![lr](./media/lr.png)


Large learning rate can prevent the gradient descent-based algorithm from convergence. If the learning rate is too big for given setup, the final estimation can just scatter around the local minimum, not being able to reach a good result. Similar behaviour was observed in our case. The probe performance was different in each run and the training loss was not decreasing.

The improvement here is modest on its own, but it compounds well with the other fixes, as described later.

### LayerNorm
As we assumed, the activations in the deeper layers of Apertus-8B-Instruct-2509
have a very large magnitude. Another way to stabilize the runs and improve
performance is to apply LayerNorm before the probe. This technique normalizes
the activations to zero mean and unit variance, ensuring that the probe input
is optimally scaled. This can yield a similar effect to using better precision —
floating point arithmetic is much more accurate near 0 than at very large
scales, so by re-centering the activations we effectively get more reliable
gradient updates for free ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)).

![ln](./media/layernorm.png)

Using LayerNorm significantly improved the probe performance, making it almost
as good as the probe trained on Llama-3.1-8B-Instruct activations. This
strongly supports our exploding activations hypothesis. If the problem were
something deeper, like missing hallucination signal in Apertus's residual
stream, normalization alone wouldn't fix it. The fact that it nearly closes the
gap with Llama suggests that Llama's activations are simply much smaller in
magnitude, and that by normalizing Apertus activations before the probe we're
bringing them into a comparable range.
### LoRA 

As described in the reproduced paper, LoRA help improving the probe performance in general. Because adapters trained accross different transformer layers slightly modify the activations to make them more aligned with the task, the probe perfomance improves:



![lora](./media/lora-apertus.png)


LoRA's benefit here is twofold. As shown in the original paper, the learned
low-rank projection aligns the activations more closely with the classification
task. But in the Apertus case, it likely also helps by compressing the
high-dimensional, high-variance activations into a more compact representation
where the probe can find a cleaner decision boundary.


### Full Solution Performance

The combined effect of all four interventions is summarized below:

| Probe model (train) | Evaluation set | Metric | Baseline mean | Full-solution mean | Absolute improvement |
|---|---|---|---:|---:|---:|
| Apertus-8B-Instruct-2509 | Apertus-8B-Instruct-2509 | AUC | 0.7025 | 0.8961 | +0.1935 |
| Apertus-8B-Instruct-2509 | Apertus-8B-Instruct-2509 | R@0.1 | 0.3837 | 0.6802 | +0.2966 |
 

The AUC improvement of ~0.19 is substantial. This puts the Apertus probe in a
range comparable to what the original paper reports for Llama-class models. The
recall at 0.1 FPR improvement (+0.30) is even more important for practical use,
since real-time hallucination detection systems typically operate under tight
false positive budgets. More than doubling recall at that threshold makes the
probe genuinely usable in a production context.

### Stability summary 

The stability gains are just as striking as the performance gains:

| Stability indicator | Baseline | Full-solution | Change |
|---|---:|---:|---:|
| Mean final training loss | 8.236 | 0.232 | 97.2% lower |
| Seed-level loss std (avg over layers) | 11.887 | 0.099 | 99.2% lower |


For context: the baseline Llama probe achieves a mean final loss of 0.480. Our
full-solution Apertus probe reaches 0.232, which is actually *lower* than the Llama
baseline, despite starting from activations that were causing 100× larger
losses. The near-elimination of seed-level variance (99.2% reduction) means
the probe training is now reliable and reproducible, which is a basic
prerequisite for any serious deployment.

## Conclusions and Future Work

In this short doc, we showcased how an *active interpretability* approach can
help in improving the performance of machine learning models. We went through a
simple observation, to problem diagnostics, setting the hypothesis and finding
the solution to the discovered problem.

What started as a routine comparison between two models turned into a
well-grounded investigation showing that Apertus-8B-Instruct-2509 has a problem
with exploding activations in its deeper layers. We successfully mitigated this
by proposing a few targeted changes to the probe training setup, and showed that
the resulting probes are not only stable but can be further improved. Crucially,
this also confirms that Apertus's residual stream does encode hallucination
signal, and that our initial approach was simply unable to extract it reliably.

Our future work will include adapting the probes for a production-wide setting.
We also look forward to verifying whether the activation behaviour changes in
upcoming Apertus releases, as fixing the root cause at the model level would
make these workarounds unnecessary.

## Acknowledgements

We would like to thank Anna Hedström for supervision throughout the project, and
Julian Minder for valuable discussions about the activation instability
hypothesis and potential fixes.






