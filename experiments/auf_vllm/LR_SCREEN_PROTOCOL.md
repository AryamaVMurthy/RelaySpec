# Learning-rate screen, fixed before its results

Use512 nested calibration records,3epochs,seed42, and learning rates
1e-4/3e-4/1e-3 for each of ZIP, uniform CE, and AUF. Token arms share4 records
per microbatch,4 anchors per record,8 accumulation steps, optimizer, and
schedule. ZIP retains its original feature sampling/weighting; its update
count and GPU time differ and must be reported.

The selection metric is mean accepted correct prefix at the fixed third-epoch
checkpoint, measured at four fixed teacher-forced anchors on each of the
separate1024 offline-validation records. Select the smallest learning rate
within1% of that arm's best prefix mean. This is an offline proxy selection,
not a throughput result or a formal significance test. No benchmark decoding
answers enter this selection. Preserve all nine outcomes.

Confirm the selected setting at4096 records and then on the declared128-request,
2048-token decoding endpoints. Existing4096/lr1e-4 token fits and the ZIP1e-3
reference remain documented initial configurations. Do not relabel the512-record
screen as the main data-scaling experiment; its purpose is optimizer selection.
