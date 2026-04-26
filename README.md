# 🍷 MLflow on AWS — Wine Quality Prediction

A end-to-end MLOps project that trains an **ElasticNet regression model** on the Wine Quality dataset, tracks experiments with **MLflow**, and stores artifacts remotely on **AWS S3** — with the MLflow Tracking Server hosted on an **EC2 instance**.

---

## 📌 Project Description

This project demonstrates how to integrate MLflow with AWS infrastructure for scalable, cloud-backed ML experiment tracking. It trains a scikit-learn ElasticNet model to predict wine quality, logs parameters and metrics to a remote MLflow server running on EC2, and stores model artifacts in an S3 bucket.

---

## 🗂️ Project Structure

```
mlflow-with-AWS/
├── app.py               # Main training script with MLflow tracking
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata
├── .python-version      # Python version pin
├── uv.lock              # Lockfile for uv package manager
└── .gitignore
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | ElasticNet model training |
| MLflow | Experiment tracking & model registry |
| AWS EC2 | MLflow Tracking Server hosting |
| AWS S3 | Remote artifact storage |
| AWS CLI / boto3 | AWS interaction |
| pipenv | Python environment management |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/navneetsxngh/mlflow-with-AWS.git
cd mlflow-with-AWS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ☁️ AWS Setup

### Step 1 — IAM User

1. Log in to the [AWS Console](https://console.aws.amazon.com/).
2. Create an IAM user with **AdministratorAccess**.
3. Configure credentials locally:
   ```bash
   aws configure
   ```

### Step 2 — S3 Bucket

Create an S3 bucket to store MLflow artifacts:
```bash
aws s3 mb s3://your-mlflow-bucket-name
```

### Step 3 — EC2 Instance (MLflow Tracking Server)

1. Launch an **Ubuntu EC2** instance.
2. Add an inbound **Security Group rule** to allow traffic on port `5000`.
3. SSH into the instance and run:

```bash
sudo apt update
sudo apt install python3-pip pipenv virtualenv -y

mkdir mlflow && cd mlflow

pipenv install mlflow awscli boto3
pipenv shell

# Configure AWS credentials on EC2
aws configure

# Start the MLflow tracking server with S3 as artifact store
mlflow server -h 0.0.0.0 --default-artifact-root s3://your-mlflow-bucket-name
```

4. The MLflow UI will be accessible at:
   ```
   http://<your-ec2-public-dns>:5000
   ```

---

## 🧪 Running the Training Script

Set the remote MLflow tracking URI in your local terminal:

```bash
export MLFLOW_TRACKING_URI=http://<your-ec2-public-dns>:5000/
```

Run the training script with optional `alpha` and `l1_ratio` hyperparameters:

```bash
# Default hyperparameters (alpha=0.5, l1_ratio=0.5)
python app.py

# Custom hyperparameters
python app.py 0.3 0.7
```

### Logged Metrics

| Metric | Description |
|---|---|
| `rmse` | Root Mean Squared Error |
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `r2` | R² Score |

---

## 📊 MLflow Tracking UI

Once the server is running, open your browser and visit:

```
http://<your-ec2-public-dns>:5000
```

Here you can compare runs, visualize metrics, and manage registered models.

---

## 📦 Model Registry

The trained model is automatically registered in MLflow's Model Registry under the name **`ElasticNetWineModel`** when a remote tracking URI is configured.

---

## 📋 Requirements

- Python 3.8+
- AWS account with EC2 and S3 access
- `mlflow`, `scikit-learn`, `pandas`, `numpy`, `boto3`

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).