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

## Architecture

🏛️ The overall architecture of the federated learning system, illustrating the communication and data exchange between the central server and clients. Refer to the provided diagrams for a better understanding.

## Examples

🔍 Explore additional examples to demonstrate various use cases and functionalities supported by the repository. Examples include code snippets, dataset usage, and model training and aggregation demonstrations.

| Method          | Accuracy | Training Time |
|-----------------|----------|---------------|
| **XYZ Algorithm** | **95%**  | **2 hours**   |
| ABC Algorithm   | 92%      | 3.5 hours     |
| PQR Algorithm   | 89%      | 1.5 hours     |

### XYZ Algorithm

Description of the XYZ Algorithm, its advantages, and how it can be used effectively.

### ABC Algorithm

Description of the ABC Algorithm, its key features, and any specific use cases where it performs well.

### PQR Algorithm

Explanation of the PQR Algorithm, including its limitations and potential applications.

## Contribution

🤝 Contributions to the project are welcome! Please follow the guidelines below:

- Report issues: Use the issue tracker to report any bugs or problems you encounter.
- Feature requests: Submit feature requests to suggest improvements or new functionalities.
- Code contribution: Fork the repository, make your changes, and submit a pull request for review.

## License

⚖️ This project is licensed under the [MIT License](LICENSE.md). Make sure to review the license terms and conditions.

## Maintainers

👥 For any inquiries or support, feel free to reach out to the maintainers:

- [Maintainer Name](mailto:maintainer@example.com) - [GitHub](https://github.com/maintainer)

## Acknowledgements

🙏 Special thanks to the following individuals and organizations for their contributions and inspiration:

- [Name/Organization](https://github.com/example) - Description of their contribution or inspiration

Make sure to acknowledge anyone who has significantly contributed to your project.