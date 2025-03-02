
# Machine Learning Generic

This is a generic machine learning project made for the specific purpose of having a base structure to attempt different techniques and furthering my knowledge on the subject.

## Explanation of the Pipelines

- **train**:
  - The `train` container runs a pipeline that trains the model and generates:
    - A `.pkl` file of the trained model.
    - Preprocessor `.pkl` file that contains configuration for data intended for prediction after training.
    - Two CSVs (`test.csv` and `train.csv`) derived from the `data.csv` file located in the `artifact` folder.
  
- **predict**:
  - The `predict` container runs a simple Flask application with an HTML form. The form allows for manual data input that is then predicted according to the latest trained model.
  - The default URL for the `predict` service is `127.0.0.1:5000`.

## Project Structure

```bash
project/
├── app.py                       # Flask app for prediction
├── src/
│   ├── pipeline/
│   │   ├── train_pipeline.py    # Training pipeline script
│   │   └── predict_pipeline.py  # Prediction pipeline script
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

### Recommended

- **Docker**
- **Docker Compose**

### Other

- **Python 3.8+**
- **Conda**

> **Note**: The project can be run with or without Docker, but it is strongly recommended to run the Dockerized application as it minimizes setup time and execution to just 2 or 3 commands. Also prevents the need to install the languages or tools locally on your machine.

## Setup Instructions

### Running with Docker

1. **Clone the project**:

   ```bash
   git clone https://github.com/guilhermepsch/machine-learning-generic.git
   cd machine-learning-generic
   ```

2. **Build Docker containers**:
   On the root folder of the project, run the following command to build both the `train` and `predict` containers:

   ```bash
   docker compose build
   ```

3. **Run the containers**:
   You can now test each service individually:

   - To run the training pipeline:

     ```bash
     docker compose up train
     ```

   - To run the prediction pipeline (Flask app):

     ```bash
     docker compose up predict
     ```

### Running Without Docker (Non-Dockerized Setup)

If you prefer to run the project without Docker, follow these steps:

1. **Clone the project**:

   ```bash
   git clone https://github.com/guilhermepsch/machine-learning-generic.git
   cd machine-learning-generic
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

   - To start the Flask application (prediction service), run:

     ```bash
     python app.py
     ```

## TLDR

- The **`train`** pipeline generates a trained model and saves it in the artifact folder.
- The **`predict`** pipeline runs a simple web server (Flask) to make predictions on manually input data via an HTML form.
- It is recommended to use Docker for ease of setup, as it handles all dependencies and environments internally.

## Troubleshooting

If you encounter issues, make sure that:

- Docker and Docker Compose are correctly installed.
- You’re in the correct directory when running the commands.
- For non-Dockerized setup, ensure that the correct Python version is used and that dependencies are properly installed in the virtual environment.
