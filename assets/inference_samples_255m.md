# My-MiniGPT 254.5M Inference Samples

These are short qualitative samples from `runs/minigpt-255m/latest.pt` at training step 50,950. They are included as qualitative evidence, not as a benchmark. Sampling used temperature `0.75`, top-k `50`, top-p `0.9`, repetition penalty `1.08`, and a maximum of 120 new tokens.

## English literary continuation

**Prompt:**

> The last train left the station at midnight, but

**Continuation:**

> was never seen again. There were some fine old buildings on the road to the city; they had a gracious air of magnificence; there was no one to be seen in the crowd.

**Prompt:**

> In the quiet library, she opened a letter that had been waiting for twenty years.

**Continuation:**

> She read it with interest and asked herself: “What does it mean to you, my dear? Would you like to see me?”

## Simplified-Chinese literary continuation

**Prompt:**

> 黄昏时分，江南小镇的石板路刚被雨水洗过，

**Continuation:**

> 张恪顺着水流向上爬了过来。路边放着一只手枪，张恪下车，陈静也下了车。

**Prompt:**

> 月光落在荒废的庭院里，墙角的桂花树还记得多年前的秋天。

**Continuation:**

> 井九来到这栋楼前，看着那处庭院里的景色，沉默了片刻后说道：“我也知道，今天的事情很复杂，所以我准备回去了。”

The samples show that the checkpoint has learned bilingual continuation and literary surface patterns, while long-range consistency and style control remain active development targets.
