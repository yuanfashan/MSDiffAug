# MSDiffAug

MSDiffAug is a two-stage representation learning framework for mass spectrometry data. It synergizes a generative diffusion augmentation strategy with a contrastive learning mechanism to effectively decouple molecular structural features from environmental noise, thereby acquiring robust low-dimensional chemical embeddings.

---





## Environment Setup

Please follow the steps below to create and activate the Conda virtual environment named `MSDiffAD` using the project's included `environment.yml` file.

### 1. Create the Virtual Environment
Open the Anaconda Prompt and run the following command to start the installation:
```bash
conda env create -f environment.yml
```



### 2. Activate the Virtual Environment
Once the environment setup is complete, activate it using the following command:
```bash
conda activate MSDiffAD
```

### 3. Verify the Installation
You can run the following command to check if PyTorch 2.2.1 has been successfully installed and is functioning correctly:
```bash
python -c "import torch; print(torch.__version__)"
```
