# FedSPA
Code of FedSPA: Generalizable Federated Graph Learning under Homophily Heterogeneity (CVPR 25)
## Abstract

Federated Graph Learning (FGL) has emerged as a solution to address real-world privacy concerns and data silos in graph learning, which relies on Graph Neural Networks (GNNs). Nevertheless, the homophily level discrepancies within the local graph data of clients, termed homophily heterogeneity, significantly degrade the generalizability of a global GNN. Existing research ignores this issue and suffers from unpromising collaboration. In this paper, we propose FedSPA, an effective framework that addresses homophily heterogeneity from the perspectives of homophily conflict and homophily bias. In the first place, the homophily conflict arises when training on inconsistent homophily levels across clients. Correspondingly, we propose Subgraph Feature Propagation Decoupling (SFPD), thereby achieving collaboration on unified homophily levels across clients. To further address homophily bias, we design Homophily Bias-Driven Aggregation (HBDA) which emphasizes clients with lower biases. It enables the adaptive adjustment of each client contribution to the global GNN based on its homophily bias. The superiority of FedSPA is validated through extensive experiments.

## Citation

``` latex
@inproceedings{fedspa_cvpr25,
  title={FedSPA: Generalizable Federated Graph Learning under Homophily Heterogeneity},
  author={Tan, Zihan and Wan, Guancheng and Huang, Wenke and Li, He and Zhang, Guibin and Yang, Carl and Ye, Mang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={15464--15475},
  year={2025}
}
```
