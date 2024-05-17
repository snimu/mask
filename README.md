# mask

What if we don't just use a causal mask forward in time (fw), but with a probability $p_{bw}$ a causal mask in the backward direction (bw)?

I can see three potential benefits from this:

1. Forcing the model to learn to distinguish between a fw and bw causal mask, and adjust properly, 
    may serve as a form of dataset augmentation that effectively increases the data diversity
2. If your limiting factor for computation is data ingestion (like I've read it is for Cerebras, for example),
    ingesting the data once, then calculating the loss on both the fw and bw mask could lead to higher 
    utilization of your accelerator
3. A ColBERT replacement. That uses BERT-style masking, but the best open models use forward masking.
    I thought it might make sense to enable those powerful models to also do the bw masking necessary to take,
    for example, footnotes into account

I have tested point 1 for now, and the results, while negative at the scales I trained at, show a promising scaling trend.
I suspect that this actually works well!


## Method

I trained hlb-gpt v0.4.1 for 10 epochs on wikitext. I predict using a fw mask and, with probability $p_bw$,
also predict using a bw mask on the same tokens (with the labels shifted etc., of course).
I accumulate the losses and then do a backward pass.
Inspired by [Yi Tay et al.'s UL2](https://arxiv.org/abs/2205.05131),
I've informed the models of their task by giving them a special token at the beginning of text
if I'm fw masking or a different one at the end of the text if I'm bw masking.
In early testing, this improved performance noticeably.

I do this for different model sizes:
I trained three models for every combination of depth in $\{4, 8, 16, 32\}$
and width in $\{192, 384, 768, 1536\}$.
This is done for different values of $p_{bw}$: $0.0$, $0.1$ and $0.05$
(in early testing, higher probabilities made the bw prediction too easy).

After a model is trained for 10 epochs, I remove the transformer layers one by one, starting from the back,
and evaluate the resulting model after each removal. I hope that this gives me some insights into
which parts of the model are more important for the fw and bw predictions, respectively. 
My first hypothesis was that early layers are used to distinguish between the fw and bw mask,
and thus the positive effects of dual masking can only occur in deep networks
that can make use of the knowledge extracted in the early layers.

Per setting (depth, width, $p_bw$, or using fw mask only), I train three times.
I've saved all results, but below, will only show you the average over all three runs for each setting.

## Results

Let's quickly work through points 2 and 3:

- Point 2: Fixing data ingestion issues (if they are even a real thing) is unlikely to happen this way.
    $p_{bw} = 10\%$ is already much worse than $p_{bw} = 5\%$, so backward masking at a significant level is undesireable.
- Point 3: Post-training models to serve as late-interaction RAG models may or may not work;
    I haven't tried yet (I'm doing this in my freetime with my own money, of which I only have limited amounts).
    Maybe I will try next month, maybe not.

With that out of the way, I will now write about point 1, improving performance given a constant number of training tokens.

### Background info

First, a caveat: I always used both a fw- and a bw-mask for the same tokens, whenever I did use a bw-mask.
I think that it's possible that the models would perform better if the fw- and bw-mask were exclusive,
but I went into the experiments with point 2 (data ingestion) in mind and finished those experiments.
I may do those other experiments at some point, if my budget allows it.

Unless specifically stated otherwise, $p_{bw} = 5\%$ for all of them, because with $p_{bw} = 10\%$,
the relative performance between models trained with the fw-mask only and those trained with a bidirectional mask
was very skewed towards the fw-only models.

### Metrics

Besides the obvious metrics, like validation loss or accuracy, I look at two ratios that are very telling:

1. **ratio ($p_{bw} = 0\%$) / ($p_{bw} = x>0\%$); \<metric\> fw**: 
    The models trained with $p_{bw} > 0\%$ are obviously better at the bw prediction
    than the ones trained with $p_{bw} = 0\%$.
    What I'm interested in here is how a non-zero $p_{bw}$ impacts the fw performance of the models.
    This *ratio* is the performance for $p_{bw} = 0\%$ divided by the performance for $p_{bw} = x\%$,
    where $x$ is usually $5$, and the performance is usually just measured by the validation loss.
2. **ratio \<metric\> fw / bw; $p_{bw} = x \ge 0 \%$**:
    How much better is a model at the fw task than the bw task?
    Lower obviously means better fw, while higher means better bw performance.
    This will be interesting when looking at performance with layers removed.

### Performance for different widths

To get a feel for how training evolves for different $p_{bw}$ at different model scales,
let's first look at the fw and bw validation loss for the default model size in hlb-gpt:

![fw & bw validation loss over epoch for original model size](results/images/fw_bw_mask_initial_backward_prob=0.05_depth=8_width=384.png)

As you can see, the fw loss is lower in the model trained only on the fw task
than for the one trained on both the fw and the bw task, even after $10$ epochs.
As we will see later, this likely changes with scale.

It is also very obvious, however, that even a small $p_{bw}$ of $5\%$ will lead to a very loss bw loss,
while the bw loss actually increases for the models trained with $p_{bw} = 0\%$.
This is a nice validation that something is working as intended,
and the intended benefits for downstream RAG tasks may actually come true.

Now, let's look at how this behaves for different model scales.


### Performance by model scale

Below, I show the ratio 1 between the fw performance of models trained with $p_{bw} = 0\%$ compared to those trained with $p_{bw} = 5\%$ (as described [above under Metrics](#metrics)) for models of different
width and depth, as well as over the combined number of parameters.

I compute the ratio independently for each of the training runs per setting,
and independently for each of $500$ steps over the approximately $10$ epochs
(for details, see the code provided in *plot_results.py*).

Then, take the ratios falling into the range $\left[\mathrm{epoch}_{\mathrm{start}}, \mathrm{epoch}_{\mathrm{stop}}\right]$
and plot them as a boxplot or violinplot.

Below, you can see the violinplot for all available data ($\mathrm{epoch}_{\mathrm{start}} = 0, \mathrm{epoch}_{\mathrm{stop}} = \mathrm{inf}$):

![(Violinplot): Ratio 1 by model size: all epochs](results/images/violinplot_ratio_by_num_params_val_losses_epoch_start_0_stop_None.png)

A few thoughts:

- There seems to be an inverse scaling law for small model sizes.
    This has its minimum at $106.8$ million parameters.
- Beyond that, positive scaling laws apply.
    As the number of parameters grows, the ratio increases as well,
    meaning that the models trained with $p_{bw} = 5\%$ are catching up in fw performance
    to the models trained with $p_{bw} = 0\%$.
- In the largest model, performance has caught up.
    This is fantastic! It implies that with the model sizes common today, using $p_{bw} = 5\%$
    at worst doesn't negatively impact fw performance, while unlocking BERT-like capabilities,
    and at best positively impacts fw performance.
- This scaling law&mdash;first inverse, then normal&mdash;applies more strongly to the width than to the depth.

So how does this apply at different epochs during training?
Here is the violinplot of ratio 1 for only the first epoch
($\mathrm{epoch}_{\mathrm{start}} = 0, \mathrm{epoch}_{\mathrm{stop}} = 1$):

![(Violinplot) Ratio 1 by model size: epoch 1](results/images/violinplot_ratio_by_num_params_val_losses_epoch_start_0_stop_1.png)

It looks like in the first epoch, even the largest model has a ratio significantly below $1$.
This implies that it takes many samples for the models trained with $p_{bw} = 5\%$ to catch up to those
trained with $p_{bw} = 0\%$ in fw performance.

To check out how the performance looks after the models have been trained for a bit, let's look at the ratio for epochs 5 to 10.
This means that the poor performance at early training phases doesn't impact the statistics,
and we get a better idea of what the models will converge to.
Here is ($\mathrm{epoch}_{\mathrm{start}} = 5, \mathrm{epoch}_{\mathrm{stop}} = \mathrm{inf}$)

![(Violinplot) Ratio 1 by model size: epochs 5 to 10](results/images/violinplot_ratio_by_num_params_val_losses_epoch_start_5_stop_None.png)


Observations:

- The scaling effects are much stronger here!
- In the largest model, the median ratio is essentially 1.

Taken together, this makes me think that with more scaling, models trained with $p_{bw} = 5\%$ could have the same or even better
fw performance as ones trained with $p_{bw} = 0\%$.

### Performance by model layer

Let's cut away the model's layers one by one and evaluate every time.

#### Ratio 1: $p_{bw} = 0\%$ / $p_{bw} = 5\%$; fw task

How does the relative performance of the models trained with $p_{bw} = 0\%$ vs those trained with $p_{bw} = 5\%$ change
over the layers?

Note that I only compare the performance of the final checkpoint here, so this is much less statistically meaningful than
the analysis above, where I compared performance for many steps throughout training.
However, it gives an indication of the trends in the models per layer and model size.

![fw-val-loss for p_bw=0% vs p_bw=5%](results/images/fw_cut_losses_with_fw_vs_bidirectional_mask_over_number_of_layer_remaining.png)

Note: while the ratio is often $1.0$ in this plot, that is because I round to one significant digit, and a lot of the ratios are merely close to $1.0$. To make this more clear, here is the same plot, but for the validation perplexity instead of the loss:

![fw-va-pplxs for p_bw=0% vs p_bw=5%](results/images/fw_cut_pplxs_with_fw_vs_bidirectional_mask_over_number_of_layer_remaining.png)



#### Ratio 2: fw / bw task; $p_{bw} = 5\%$

How does the relative performance between the fw and bw task change?
Importantly, this is not a comparison of fw performance for two different values of $p_{bw}$ as before,
but a comparison between fw and be performance for the same value of $p_{bw} = 5\%$.

![fw- vs bw-perf for p_bw=5%](results/images/fw_vs_bw_perf_with_bidirectional_mask_over_number_of_layer_remaining_cut_accs.png)

In the image above, I plot the ratio of fw- to bw-performance for models trained with $p_{bw} = 5\%$
over the number of layers used, for different numbers of parameters.
The numbers shown are always the average over 3 runs.

A few observations:

1. As the number of layers is cut more and more, the performance of the bw task compared to the fw task drops off rapidly.
    This implies to me that the model learns the fw task, and performs that in the early layers,
    and then somehow manages to invert it into the bw task in the later layers.
2. In the late layers&mdash;and especially in the late layers of the fairly deep networks&mdash;we see that the bw task
    is much easier for the model than the fw task.
3. The change from being better at the bw than the fw task goes in a sigmoid-like fashion starting from the first third or half
    of the model until the very end (mostly, this is just eyeballing the plots). In the shallow models, this of course looks
    pretty sudden.

To be clear, the absolute performance on both the fw and the bw task falls rapidly with every layer that is removed.
However, as a trend, the later layers seem more important for the bw task, and the earlier layers for the fw task.


### Future experiments

- Train an even larger model to see if we actually get a positive effect on the fw performance
    from training bidirectionally.
- Instead of the choice being between training on either both the fw and bw task or only the fw task,
    make it a choice between training on just the fw or just the bw task.
- Finetune some open LLM (phi3 or whatever) on a dataset with this method,
    then train it to perform as a ColBERT-replacement RAG tool.

I'm not sure if I will actually get to any of those;
I have a lot of other ideas I want to explore, and way to little money to do it all.



## Acknoledgements

As always, mostly based on [Fern](https://github.com/tysam-code)'s [hlb-gpt](https://github.com/tysam-code/hlb-gpt).

```
cff-version: 1.2.0
message: "Citations would be appreciated if you end up using this tool! I currently go by Fern, no last name given."
authors:
  given-names: "Fern"
title: "hlb-gpt"
version: 0.4.0
date-released: 2023-03-05
url: "https://github.com/tysam-code/hlb-gpt"
```
