# Ornith-1.5-122B expert-usage profiles

How often the router picked each routed expert, per layer (the MTP block last), counted by the engine with
`STRATA_Q36_PROFILE_OUT` over 18 agent-style prompts (build logs, engine logs, source code; about 34,000 tokens plus 64
generated tokens each), on UD-Q4_K_XL split over two RTX 3090s. Setup passes `data/ornith/<size>.profile` as
`STRATA_Q36_PROFILE`, and the engine fills each layer's VRAM slots in this order at start-up.

The router weights are fp32 in every Ornith file, so the counts describe the model's routing rather than one quant;
`UD-IQ2_XXS.profile` is the same file. Print the usage curve with `python3 tools/ornith/profile_curve.py <file>`.
