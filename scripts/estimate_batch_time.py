"""Estimate total runtime for ablation batch."""
batches_per_epoch = {(96, 384): (384-4)/16 * 0.015}  # rough: ~0.57s/epoch
# More carefully: each fold trains on N-1 subjects
# ADSZ: 96 subs, 4 trials/sub → train=380 trials → 24 batches/epoch × 20 epochs × 5ms = 2.4s per epoch
# But actually each fold rebuilds model + trains + evaluates test → ~4s/fold

datasets = [
    ("ADSZ", 96, 384),
    ("ADFSU", 92, 460),
    ("ds004504", 65, 23304),
]
print("Dataset    Subjects  Trials    Folds   Epochs   ~sec/fold  ~min/variant  ~h/10var×3seed")
print("-" * 90)
for name, n_sub, n_trial in datasets:
    trials_per = n_trial / n_sub
    train_trials = n_trial - trials_per
    batches_per_epoch = train_trials / 16
    # Each epoch: forward/backward on train + validation
    # RTX 4060 ~5ms per batch for this model size (1.6M params)
    ms_per_batch = 5
    sec_per_epoch = batches_per_epoch * ms_per_batch / 1000 * 2  # ×2 for val pass
    avg_epochs = 8  # early stopping typically stops around 6-10
    sec_per_fold = sec_per_epoch * avg_epochs + 1.5  # +1.5s for model build + eval
    min_per_variant = n_sub * sec_per_fold / 60
    total_hours = min_per_variant * 10 * 3 / 60
    print(f"{name:12s} {n_sub:5d}     {n_trial:6d}    {n_sub:5d}   {avg_epochs:3d}     {sec_per_fold:6.1f}      {min_per_variant:6.1f}         {total_hours:5.1f}")
