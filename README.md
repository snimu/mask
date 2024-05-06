# mask

What if we don't just use a causal mask forward in time (fw), but with a probability $p_{bw}$ a causal mask in the backward direction (bw)?

I can see three potential benefits from this:

1. Forcing the model to learn to distinguish between a fw and bw causal mask, and adjust properly, 
    may serve as a form of dataset augmentation that effectively increases the data diversity
2. If your limiting factor for computation is data ingestion (like I've read it is for Cerebras, for example),
    injesting the data once, then calculating the loss on both the fw and bw mask could lead to higher 
    utilization of your accelerator
3. A ColBERT replacement. That uses BERT-style masking, but the best open models use forward masking.
    I thought it might make sense to enable those powerful models to also do the bw masking necessary to take,
    for example, footnotes into account

All in all, these experiments are mostly a failure, but I'm going to document the results here anyways.


## Method

I trained hlb-gpt v0.4.1 for 10 epochs on wikitext. I predict using a fw mask and, with probability $p_bw$,
also predict using a bw mask on the same tokens (with the labels shifted etc., of course).
I accumulate the losses and then do a backward pass.

I do this for different model sizes (specifically, I control both the depth and width of the model),
and to values of $p_{bw}$: 0.1 and 0.05 (in early testing, higher probabilities made the bw prediction too easy).

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

- Point 2: Fixing data ingestion issues (if they are even a real thing) isn't going to happen this way.
    $M_{0.1}$ is already much worse than $M_{0.0}$, so backward masking at a significant level is undesireable.
- Point 3: Post-training models to serve as late-interaction RAG models may or may not work;
    I haven't tried yet (I'm doing this in my freetime with my own money, of which I only have limited amounts).
    Maybe I will try next month, maybe not

With that out of the way, I will now write about point 1, improving performance given a constant number of training tokens.

TODO: get more out of the data you have???




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
