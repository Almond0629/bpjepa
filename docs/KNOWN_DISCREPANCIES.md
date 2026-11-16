# Known discrepancies

The release preserves original behavior. Nothing below was changed to make it agree with the paper's text.

## Behavior that looks odd but is preserved

1. **Downstream RNG differs between JEPA arms and COMEDIAN.** The JEPA initializer builds its activation-check model on the main RNG stream during model construction; the COMEDIAN route forks the RNG for that check. Fresh temporal and classifier weights are identical across arms, since they are drawn earlier, but the training RNG stream after construction is not. The paper's statement that head initialization and data order are identical across arms refers to the frozen evaluation, where it holds.
2. **Probe masks are tied to torch 2.0.1.** `torch.randint` on a seeded CPU generator gives different integers under torch 2.0.1 and 2.11. The stored masks reproduce the sampler under 2.0.1 (checked bitwise), and the paper's GPU extraction replayed them under 2.11. They ship as `configs/probes/probe_masks.npz`. The same fact means pretraining masks under torch 2.0.1 differ from the paper's four-arm runs under torch 2.11.
3. **Decoder provenance.** Historical downstream and probe jobs bound the `PIL` of the COMEDIAN environment before anything else, to keep decoded pixels bitwise equal to earlier runs. Both recorded environments now report Pillow 12.2.0, but the decoder builds may differ (conda vs wheel), so pixels on another stack are not guaranteed to be bitwise equal. Fig. 2 shows this: `bpjepa-fig2` reproduces the paper's Fig. 2 with 0 differing pixels only when Pillow is built against libjpeg 9 (a conda Pillow, as used for the paper). With pip wheels (libjpeg-turbo), as installed by the documented stack D, the same command gives a PNG of the same size in which 6.24% of pixels differ (at most 24 per channel), confined to the decoded video frames and heat-map backgrounds; `fig2_evidence.json`, the data, is identical.
4. **Probe statistics depend slightly on the torch version.** Re-analysing the same features on CPU gives values that differ between torch 2.0.1 and torch 2.11 by at most 8.2e-7 absolute, in the sections that depend on the head fit. Two environments on torch 2.11 with NumPy 1.26 and 2.2 give identical values, so NumPy is not the cause; the cause is not confirmed. Printed table values are unchanged.
5. **Historical names in result files.** Result files keep historical keys (`R0`, `EB0`, `U`, `EB`) and wording such as "readout" in their text fields, since they are records.

6. **ECC alignment is not bit-reproducible across machines.** On one half regenerated on another machine, 8 of 5,399 `alignment_status` values and 39 `n_valid` values differ from the stored measurements (OpenCV 5.0.0 on both), while `states` and `reasons` match on every transition. Original and release code give bitwise equal measurements on the same machine (first 300 frames). So evidence regenerated on other hardware can differ slightly from the paper's target tables, whose array hashes are in `configs/evidence/identities.json`. This depends on the machine, not the code. On the machine that produced the stored artifacts, regeneration was bit-exact on 6,380 transitions of 20 halves. On a second machine with a different CPU, the same release code differed in 6 `alignment_status`, 9 `ecc_status` and 44 `n_valid` entries of those 6,380, with `cut`, `wipe`, `motion_bin`, `states` and `reasons` all equal. Downstream, raw change differed by at most 8.7e-3 where mask and alignment agree, and the supported and eligibility arrays and clip validity were unchanged.

7. **Frozen features depend on the stack.** Release and original cache code are bitwise equal on one stack. Against the paper's GPU cache, CPU features differ by at most about 4e-4 under torch 2.11 but by about 0.03 under torch 2.0.1, on the same frames and weights. So build caches on the paper's stack (P+D), and do not expect another stack to match. The historical seeds 203/666 used a cache built on the RTX 5090 stack with their heads trained on RTX A6000s.

## Fixed in packaging (no scientific effect)

8. The historical trainer crashed after the final checkpoint when exporting with the metadata dataset (`dataset[0]`), and exports were recovered separately on CPU. The release exports from the underlying dataset directly.
9. `--frame-root` replaces absolute frame paths in manifests.

## Not verified

No full training run was repeated from this repository, so the paper's numbers are not re-derived from new training, and reproduction on other hardware is not established.
