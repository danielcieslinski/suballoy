# Suballoy: Crystal Graph Neural Networks for Substitutional Alloying
Welcome to the official repository of **Substitutional Alloying Using Crystal Graph Neural Networks**. This repository contains the code used in the paper authored by _Dario Massa_, _Daniel Cieslinski_, _Amirhossein Naghdi_, and _Stefanos Papanikolaou_.

## Table of Contents

- [Suballoy: Crystal Graph Neural Networks for Substitutional Alloying](#suballoy-crystal-graph-neural-networks-for-substitutional-alloying)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
  - [Data Preparations](#data-preparations)
    - [Data Redistribution Notice](#data-redistribution-notice)
    - [Downloading Data from The Materials Project](#downloading-data-from-the-materials-project)
    - [Reproducing our results](#reproducing-our-results)
  - [Producing Plot Data](#producing-plot-data)
  - [Generating Plots](#generating-plots)
  - [License](#license)
  - [Citation](#citation)

## Environment Setup

This project was developed using Python 3.7.13. To ensure that you have the correct Python version, it is recommended to create a virtual environment. 

```shell
conda create -n suballoy-env python=3.7.13
conda activate suballoy-env
pip install -r requirements.txt
```

The periodic table plots require an additional dependency, a webdriver. You can install the geckodriver using Conda as follows:

```shell
conda install -c conda-forge firefox geckodriver
```

Note: The geckodriver requires Firefox to be installed on your system. Make sure it's installed before proceeding.

## Data Preparations

### Data Redistribution Notice

Please note that due to the data redistribution policies of The Materials Project, we are not able to redistribute the Materials Project dataset in its original form. However, in the `res/` directory, you can find pickled data that will allow you to recreate all the plots included in the paper.

### Downloading Data from The Materials Project

If you wish to reproduce plots using the latest data available on The Materials Project, please follow these steps:

1. Obtain your Materials Project API key by following the instructions [here](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started).

2. Paste your API key into a file named `mat_proj_api_key.txt` inside the `.secrets/` directory.

3. Run the following command to download the data:

```shell
python mat_proj.py
```

### Reproducing our results

If you wish to reproduce our results exactly, please filter the downloaded dataframe (`res/mp_df.pickle`) with the IDs that we used to create our plots. You can find the list of IDs we used in our dataset at `res/mp_id_list.csv`.

## Producing Plot Data

After downloading the data, run the following command to process the data necessary for the plots. This command will recreate all files included in the `res/` directory.

```shell
python produce_plot_data.py
```

## Generating Plots 

To regenerate all the plots featured in the paper, run the following command:

```shell
python make_plots.py
```

This will create plots and save them in a designated directory.

## License

This project is licensed under the MIT License. See `LICENSE` file for more details.

## Citation

If you use the code or data from this repository in your research, please cite our paper:

```
https://doi.org/10.48550/arXiv.2306.10766
```

For further inquiries or if you need assistance, please don't hesitate to open an issue or contact the authors. Enjoy exploring substitutional alloying with crystal graph neural networks!
