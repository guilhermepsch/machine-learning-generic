
# Machine Learning Generic

This is a project made for the specific purpose of trying for a position as a Junior Deep Learning Engineer.

## Explanation of the Pipeline

- **train**:
  - The `train` container runs a pipeline that trains the model and generates:
    - A `.pkl` file of the trained model.
    - Preprocessor `.pkl` file that contains configuration for data intended for prediction after training.
    - Two CSVs (`test.csv`, `train.csv` and `data.csv`) derived from the `data.pkl` file located in the `artifact` folder.
    - Several images in `.png` format that show visualization of data, and evaluations of both trained models.

## Project Structure

```bash
project/
├── src/
│   ├── pipeline/
│   │   └── train_pipeline.py    # Training pipeline script
│   ├── components/              # Steps to train the model, separated by usage
│   ├── logger.py                # Logging configuration
│   ├── exception.py             # Custom Exception class
│   └── utils.py                 # Functions such as train evaluation, load and save objects.
├── Dockerfile                   # Dockerfile for containerizing the application
├── docker-compose.yml           # Docker Compose configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Prerequisites

Before running the project, make sure you have the following installed on your machine:

### **Recommended**

- **[Git](https://git-scm.com/)**
- **[Docker](https://www.docker.com/)**
- **[Docker Compose](https://docs.docker.com/compose/)**

### **Alternative**

- **[Python 3.8+](https://www.python.org/downloads/release/python-380/)**
- **[Conda](https://docs.conda.io/projects/conda/en/latest/index.html)**

> **Note**: The project can be run with or without Docker, but it is strongly recommended to run the Dockerized application as it minimizes setup time and execution to just 2 or 3 commands. Also prevents the need to install languages, tools and dependencies locally on your machine.

## Setup Instructions

### Running with Docker

1. **Clone the project**:

   ```bash
   git clone https://github.com/guilhermepsch/machine-learning-generic.git
   cd machine-learning-generic
   git checkout ml-junior-test
   ```

2. **Build Docker containers**:
   On the root folder of the project, run the following command to build the `train` container:

   ```bash
   docker compose build
   ```

3. **Run the container**:

   - To run the training pipeline:

     ```bash
     docker compose up train
     ```

### Running Without Docker (Non-Dockerized Setup)

If you prefer to run the project without Docker, follow these steps:

1. **Clone the project**:

   ```bash
   git clone https://github.com/guilhermepsch/machine-learning-generic.git
   cd machine-learning-generic
   git checkout ml-junior-test
   ```

2. **Create a virtual environment**:
   It’s recommended to create a virtual environment to avoid installing dependencies globally. You can create one using **`venv`** (if using Python) or **`conda`** (if using Conda).

   For Python 3.8+:

   ```bash
   python3 -m venv venv
   ```

   Or, using Conda (optional):

   ```bash
   conda create --name myenv python=3.8
   conda activate myenv
   ```

3. **Activate the virtual environment**:
   - If you’re using `venv`:

     ```bash
     source venv/bin/activate  # On Linux/Mac
     venv\Scripts\activate     # On Windows
     ```

   - If using Conda:

     ```bash
     conda activate myenv
     ```

4. **Install dependencies**:
   Install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the entry points**:
   Now, you can manually run the Python scripts:

   - To train the model, run:

     ```bash
     python src/pipeline/train_pipeline.py
     ```

## TLDR

- The **`train`** pipeline generates a trained model and saves it in the artifact folder.
- It is recommended to use Docker for ease of setup, as it handles all dependencies and environments internally.

## Troubleshooting

If you encounter issues, make sure that:

- Docker and Docker Compose are correctly installed.
- You’re in the correct directory when running the commands.
- For non-Dockerized setup, ensure that the correct Python version is used and that dependencies are properly installed in the virtual environment.
