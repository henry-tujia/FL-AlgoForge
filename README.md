# FL-AlgoForge

📚 **Federated Learning Repository**

A repository for implementing and reproducing classic federated learning methods. This project allows for fast training by leveraging multiprocessing to launch client processes. It supports both local training on clients and centralized server aggregation with customizable operations, making it highly extensible.

## Installation

🔧 Use the following steps to install the project:

1. Clone the repository: `git clone https://github.com/henry-tujia/FL-AlgoForge.git`
2. Navigate to the project directory: `cd FL-AlgoForge`
3. Install [hatch](https://github.com/pypa/hatch) and initialize the project: `pip install hatch&&hatch new --init`
4. Switch Python environment: `hatch shell`.
5. Install  and dependencies: `pip install -r requirements.txt`

## Quick Start

🚀 Follow the steps below to quickly get started:

1. Set the dataset directory in `configs/datasets/cifar10.yaml` (recommended for quick start with Cifar10).
2. Launch training using multiprocessing: `python main.py method=fedavg models=resnet8`.
3. Check the results in the `logs` folder or use wandb to track the results.


## Usage

📘 Detailed instructions on how to use the repository:

1. Configuration: Customize the settings by changing the YAML files in the `configs/` directory. These files contain parameters and federated settings that can be adjusted to fit your requirements.

2. Dataset Partitioning: To implement custom dataset partitioning, modify the dataset code in `src/data_preprocessing/${dataset}/data_loader.py`. This file handles the data loading process, and you can modify it to support your desired data splitting method. Currently, it supports partitioning client subdatasets based on the dataset's indices (`idxs`).

3. Algorithm Customization: To customize the algorithm, you can inherit from the `Base_Client` and `Base_Server` classes defined in `src/methods/base.py`. By inheriting from these base classes, you can make the necessary modifications to tailor the algorithm to your specific requirements.

Please refer to the respective files and directories mentioned above for detailed instructions, and make the necessary changes according to your needs.

## Performance Summary

🔍 Explore examples showcasing algorithm performance and training time. The table below presents accuracy and training time for each algorithm.

```
CONFIG
├── federated_settings
│   └── comm_round: 200 
│       client_sample: 0.1
│       client_number: 100
│       thread_number: 10
├── local_setting
│   └── lr: 0.01
│       wd: 0.0001
│       epochs: 10
│       local_valid: false
└── datasets
    └── dataset: cifar10
        batch_size: 64  
        num_classes: 10 
        partition_method: hetero
        partition_alpha: 0.1
```

| Method     | Accuracy | Training Time  |
|------------|----------|---------------------------|
| FedAvg     | 45.91%   | 8.69 seconds/round              |
| FedProx    | 38.08%   | 8.68 seconds/round              |
| MOON       | 89%      | 1.5 hours/round                 |

### FedAvg

FedAvg is a federated learning algorithm that trains models on multiple devices or machines, performs local updates on each device, and then averages the updated model parameters to achieve global model updates.

[[paper link]](https://arxiv.org/pdf/1602.05629v1/1000)

### FedProx

FedProx is a federated learning algorithm that addresses the challenges of data heterogeneity in distributed networks. It incorporates a regularization term to improve the performance of the global model by penalizing the divergence between local models and the global model during the aggregation process.
$$f_i(w) = l(x_i, y_i, w) + (\mu/2) * ||w - w_t||^2$$

[[paper link]](https://arxiv.org/pdf/1812.06127)
### MOON

The key idea of MOON is to utilize the similarity between model representations to correct the local training of individual parties. This is achieved by conducting contrastive learning in the model-level. MOON aims to maximize the agreement between the representation learned by the current local model and the representation learned by the global model. 
$$ℓ_{con} = -log(exp(sim(z, z_glob) / τ) / (exp(sim(z, z_glob) / τ) \\+ exp(sim(z, z_prev) / τ)))$$
$$ℓ = ℓ_{sup}(w_i^t; (x, y)) + \mu ℓ_{con}(w_i^t; w_i^{t-1}; w^t; x)$$

[[paper link]](https://arxiv.org/pdf/2103.16257.pdf) [[code link]](https://github.com/QinbinLi/MOON)
## Contribution

🤝 Contributions to the project are welcome! Please follow the guidelines below:

- Report issues: Use the issue tracker to report any bugs or problems you encounter.
- Feature requests: Submit feature requests to suggest improvements or new functionalities.
- Code contribution: Fork the repository, make your changes, and submit a pull request for review.

## License

⚖️ This project is licensed under the [MIT License](LICENSE.md). Make sure to review the license terms and conditions.

## Acknowledgements

🙏 Special thanks to the following individuals and organizations for their contributions and inspiration:

- [FedAlign](https://github.com/mmendiet/FedAlign) - Official repository for Local Learning Matters: Rethinking Data Heterogeneity in Federated Learning [CVPR 2022 Oral, Best Paper Finalist]

- [FedML](https://github.com/FedML-AI/FedML) - The unified and scalable ML library for large-scale distributed training, model serving, and federated learning. FEDML Launch, a cross-cloud scheduler, further enables running any AI jobs on any GPU cloud or on-premise cluster. Built on this library, FEDML Nexus AI (https://fedml.ai) is the dedicated cloud service for generative AI.

- [MOON](https://github.com/QinbinLi/MOON) - Model-Contrastive Federated Learning (CVPR 2021)