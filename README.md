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

Description of the ABC Algorithm, its key features, and any specific use cases where it performs well.
[[paper link]](https://arxiv.org/pdf/1812.06127)
### MOON

Explanation of the PQR Algorithm, including its limitations and potential applications.
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