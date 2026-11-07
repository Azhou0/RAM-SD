# Paper figure

`ram_sd_overview.png` reproduces **Figure 1** from the ACL Anthology version of:

Ziyang Zhou, Ziqi Liu, Yan Wang, Yiming Lin, and Yangbin Chen. 2026. *RAM-SD: Retrieval-Augmented Multi-agent framework for Sarcasm Detection*. ACL 2026, pages 1434-1448.

- [ACL Anthology](https://aclanthology.org/2026.acl-long.64/)
- [Source PDF](https://aclanthology.org/2026.acl-long.64.pdf)
- [DOI](https://doi.org/10.18653/v1/2026.acl-long.64)
- [arXiv preprint](https://arxiv.org/abs/2601.17002)

The image was rendered directly from PDF page 4 at 360 DPI, then cropped to the figure area. Its dimensions are 2650 x 1515 pixels. The figure's content is unchanged; the typeset caption and surrounding page text are omitted.

With the source PDF available locally as `references/paper.pdf`, the extraction command is:

```bash
pdftoppm -f 4 -l 4 -r 360 -x 160 -y 330 -W 2650 -H 1515 \
  -singlefile -png references/paper.pdf assets/ram_sd_overview
```

The reproduced figure follows the paper's [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/), as stated in the [ACL Anthology copyright terms](https://aclanthology.org/info/copyright/). The repository's [MIT license](../LICENSE) applies to the implementation.
