# TCOAT for Wind Power Forecasting

### TCOAT: Temporal Collaborative Attention for Wind Power Forecasting [[Paper](https://doi.org/10.1016/j.apenergy.2023.122502)]

---

### News

🎉🎉🎉 **TCOAT** model has been integrated
into [pyFAST: Flexible, Advanced Framework for Multi-source and Sparse Time Series Analysis in PyTorch](https://github.com/freepose/pyFAST).
The implementation details can be found [here](https://github.com/freepose/pyFAST/blob/main/fast/model/mts/coat.py).
We sincerely thank [Zhijin Wang](https://github.com/freepose) and colleagues for their valuable support during the
integration process.

---

### Abstract

Wind power serves as a clean and sustainable form of energy. However, its generation is fraught with variability and
uncertainty, owing to the stochastic and dynamic characteristics of wind. Accurate forecasting of wind power is
indispensable for the efficient planning, operation, and grid integration of wind energy systems. In this paper, we
introduce a novel forecasting method termed Temporal Collaborative Attention (TCOAT). This data-driven approach is
designed to capture both temporal and spatial dependencies in wind power generation data, as well as discern long-term
and short-term patterns. Utilizing attention mechanisms, TCOAT dynamically adjusts the weights of each input variable
and time step based on their contextual relevance for forecasting. Furthermore, the method employs collaborative
attention units to assimilate directional and global information from the input data. It also explicitly models the
interactions and correlations among different variables or time steps through the use of self-attention and
cross-attention mechanisms. To integrate long-term and short-term information effectively, TCOAT incorporates a temporal
fusion layer that employs concatenation and mapping operations, along with hierarchical feature extraction and
aggregation. We validate the efficacy of TCOAT through extensive experiments on a real-world wind power generation
dataset from Greece and compare its performance against twenty-two state-of-the-art methods. Experimental results
demonstrate that TCOAT outperforms existing methods in terms of both accuracy and robustness in wind power forecasting.
Moreover, we conduct a generality study on an additional real-world dataset from a different climate condition and wind
power characteristics. The results show that TCOAT can achieve comparable or better performance than the
state-of-the-art methods, confirming the generalization ability of TCOAT.

---

### Model Architecture

![Model Architecture](model_architecture.png)

---

### Requirements

- Python 3.10+
- PyTorch 2.0.0+

---

### Dataset

This study uses two datasets of wind power generation data and associated meteorological data with different
characteristics and challenges.

---

### Experimental Setup

Min–max scaling was applied to standardized features to normalize values within the [0, 1] range.

The dataset was partitioned chronologically at a 4:1 ratio into training and testing subsets, ensuring sufficient
training data while preserving temporal continuity.

---

### Contact

If you have any questions or suggestions, please feel free to contact me at [yuehu.xm@gmail.com](yuehu.xm@gmail.com).

---

### Citation

If you find this work useful in your research, please use the following citation formats:

**BibTeX:**

```bibtex
@article{art/apen2024/357Hu,
    author = {Hu, Yue and Liu, Hanjing and Wu, Senzhen and Zhao, Yuan and Wang, Zhijin and Liu, Xiufeng},
    title = {Temporal Collaborative Attention for Wind Power Forecasting},
    journal = {Applied Energy},
    volume = {357},
    pages = {122502},
    year = {2024},
    doi = {10.1016/j.apenergy.2023.122502},
}
```

**APA/Plain Text:**

Yue Hu, Hanjing Liu, Senzhen Wu, Yuan Zhao, Zhijin Wang, and Xiufeng Liu. 2024. Temporal Collaborative Attention for
Wind Power Forecasting. Applied Energy 357 (2024), 122502. https://doi.org/10.1016/j.apenergy.2023.122502
