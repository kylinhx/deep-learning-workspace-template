<!--
 * @Author: kylinhanx kylinhanx@gmail.com
 * @Date: 2024-12-31 16:36:19
 * @LastEditors: kylinhanx kylinhanx@gmail.com
 * @LastEditTime: 2024-12-31 22:40:33
 * @FilePath: \pipeline\README.md
 * @Description: README file
-->
# Deep Learning Workspace Template (Updating...)

## Why Build This Rep?

I have undertaken some deep learning projects on my own, and I will likely be working on many similar projects in the future. Therefore, I want to create a template to quickly get started and help me set up the code for a deep learning project more efficiently. So I have decided to create this code framework and make it open source. 

If you have any questions, please leave a comment in the issues section. For collaborations or other intents, please contact me at my email: kylinhanx@gmail.com and specify your intent. If you find this repository helpful, please give it a star. Thank you!

## Template Structure

This template is organized to provide a clear and efficient workflow for developing deep learning projects. Below is an overview of the structure:

- `data/`: This directory is for storing datasets. You can include subdirectories for raw data, processed data, or any other categorization that suits your project needs.

- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA), model prototyping, and visualization. This is where you can iteratively develop and experiment with your models.

- `src/`: Contains the main source code for your project. This includes modules for data processing, model definitions, training scripts, and utility functions.
  - `common.py`: Scripts for data cleaning, transformation, and augmentation.
  - `models/`: Directory for model architectures and configurations.
  - `train.py`: Main script for training models.
  - `evaluate.py`: Script for evaluating model performance.
  - `utils.py`: Utility functions for common tasks like logging and configuration handling.
  - `predict.py`: 

- `config/`: Configuration files for managing hyperparameters, model settings, and other project-specific configurations.

- `results/`: This is where you can save model outputs, including predictions, visualizations, and performance metrics.

- `requirements.txt`: A file listing all the dependencies and libraries required to run the project. Install these dependencies using `pip install -r requirements.txt`.

- `README.md`: This file provides an overview of the project, instructions for setup, and guidelines for usage.

## Personal Code Style

In this section, I’d like to share my personal coding style preferences:

1️⃣ Use absolute paths for all file and directory references.

2️⃣ For each training session, create a new folder under the results directory to store session-specific outputs.

3️⃣ Retain all data that might be relevant or of interest during the training process.

4️⃣ For every training session, create three subdirectories within the project folder: models, logs, and figures.

5️⃣ Clearly differentiate between training, evaluation, and prediction phases.

6️⃣ Separate model architecture definitions from individual layer configurations.

## Getting Started

To get started with this template, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine using:

   ```bash
   git clone https://github.com/kylinhx/deep-learning-workspace-template.git
   ```

2. **Set Up the Environment**: Navigate to the project directory and set up a conda virtual environment:

    ```bash
    cd deep-learning-workspace-template
    conda create -n "<your env name>" python==3.10
    conda activate "<your env name>"
    pip install -r requirements.txt
    ```
    
2. **Add Your Data**: Place your datasets in the data/ directory.

3. **Data Analyze**: Use the Jupyter notebooks in the notebooks/ folder to explore your data, perform visualizations, and conduct exploratory data analysis (EDA). Make sure your data is clean, well-prepared, and ready for training.

4. **Model Build**: Define your model architecture in the src/models/ folder. You can build custom models or use pre-trained ones. Make sure to separate the architecture and layer definitions clearly. Store model configuration settings in the config/ folder for easy modification.

5. **Modify Code**: Adjust the train.py, evaluate.py, and predict.py scripts to suit your model and dataset. Ensure that hyperparameters and other settings are parameterized through configuration files.

6. **Begin Training**: Once everything is set up, you can start training your model by running the training script:
    ```bash
    python src/train.py --config config/<your_config_file>.yaml
    ```
  You can monitor the training process using the logs generated in the logs/ subdirectory, and visualize your training progress in the figures/ subdirectory.



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=kylinhx/deep-learning-workspace-template&type=Date)](https://star-history.com/#kylinhx/deep-learning-workspace-template&Date)