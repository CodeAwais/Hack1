# MSDetect: Early Detection Web Application for Multiple Sclerosis

## Overview

MSDetect is a web application designed for the early detection of Multiple Sclerosis (MS) using MRI scans. This project was inspired by personal experiences within our families, where early detection could have significantly improved the management and quality of life for affected individuals. MSDetect aims to provide a preliminary assessment tool that can aid in identifying potential indicators of MS, encouraging timely consultation with medical professionals for comprehensive diagnosis and treatment.

**Disclaimer:** MSDetect is intended for informational purposes only and should not be used as a substitute for professional medical advice. The application provides a preliminary analysis and does not offer a definitive diagnosis. Consult with a qualified healthcare professional for accurate diagnosis and treatment.

## Features

- **MRI Scan Analysis:** Accepts MRI scans as input (DICOM, JPG, JPEG, PNG) to analyze for potential indicators of MS.
- **AI-Powered Prediction:** Employs a pre-trained ResNet50-based deep learning model to predict the likelihood of MS presence based on the uploaded MRI scan.
- **Patient Details Input:** Allows users to input additional information such as age, gender, symptoms, family history, smoking history, and Epstein-Barr virus (EBV) status to refine the analysis.
- **AI Report Generation:** Integrates with the Perplexity AI API to generate a comprehensive report summarizing the findings and providing potential risk assessments based on the MRI analysis and inputted patient details.
- **User-Friendly Interface:** Features a clean and intuitive web interface for easy navigation and data input.

## Technologies Used

- **Frontend:**
  - Next.js
  - shadcn/ui
- **Backend:**
  - FastAPI (Python framework for building APIs)
  - Python
  - PyTorch (Deep learning framework)
  - Pydicom (Library for reading DICOM files)
  - OpenCV (Library for image processing)
  - Requests (Library for making HTTP requests)
- **AI Services:**
  - Perplexity AI API (For report generation)

## Setup and Installation

**Prerequisites:**

- Python 3.7+
- Node.js 16+
- `pip` (Python package installer)
- `npm` or `yarn` (JavaScript package manager)

**Installation Steps:**

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Set up the backend:**

    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```

3.  **Configure the environment variables in the backend:**

    - Create a `.env` file in the `backend` directory.
    - Add the following variables, replacing `<your_perplexity_api_key>` with your actual Perplexity AI API key:

      ```
      PERPLEXITY_API_KEY=<your_perplexity_api_key>
      ```

4.  **Start the backend server:**

    ```bash
    uvicorn main:app --reload
    ```

5.  **Set up the frontend:**

    ```bash
    cd ../frontend
    npm install  # or yarn install
    ```

6.  **Configure the environment variables in the frontend:**

    - Create a `.env.local` file in the `frontend` directory

    - Add the following variable, replacing `<your_backend_url>` with the actual URL of your backend

      ```
      NEXT_PUBLIC_BACKEND_URL=<your_backend_url>
      ```

7.  **Start the frontend development server:**

    ```bash
    npm run dev  # or yarn dev
    ```

8.  **Access the application:** Open your web browser and navigate to the address displayed by the Next.js development server (usually `http://localhost:3000`).

## Usage

1.  **Upload an MRI Scan:** Upload an MRI scan in DICOM, JPG, JPEG, or PNG format using the file input field.
2.  **Enter Patient Details:** Fill out the form with the patient's age, gender, symptoms (if any), family history, smoking history, and EBV status.
3.  **Generate Report:** Click the "Generate Report" button to submit the data to the backend.
4.  **Review Results:** The application will display a preliminary analysis of the MRI scan, along with a generated report summarizing the findings and providing a potential risk assessment.

## Model Training

The deep learning model used in MSDetect is a modified ResNet50 architecture. The model was trained on a dataset of MRI scans labeled with MS presence. The training process involved:

- **Data Preprocessing:** Resizing and normalizing the MRI scans.
- **Model Training:** Fine-tuning the ResNet50 model using the prepared dataset.
- **Evaluation:** Evaluating the model's performance on a held-out test set.

_NOTE: Details about the dataset, the training script and the performance matrics of the training can be added to this readme._

## Future Enhancements

- Integration with medical imaging APIs for direct access to patient records (with appropriate permissions and security measures).
- Implementation of explainable AI (XAI) techniques to provide insights into the model's decision-making process.
- Support for additional MRI modalities and image processing techniques.
- Improved user interface and report generation features.
- Expanding model training to include a more diverse and larger dataset.
