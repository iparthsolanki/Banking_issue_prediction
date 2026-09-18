# Banking77 Intent Prediction using BiGRU

<p align="center">
  <img src="Banking77.jpeg" alt="Banking77 Intent Prediction" width="1000"/>
</p>

<p align="center">
  <strong>Deep Learning | NLP | BiGRU | FastAPI | HTML | CSS | JavaScript</strong>
</p>

---

## Project Overview

**Banking77 Intent Prediction** is an end-to-end Natural Language Processing and Deep Learning application designed to understand customer banking queries and automatically classify them into one of **77 predefined banking intents**.

The project uses the **Banking77 dataset** and a custom **Bidirectional GRU (BiGRU)** neural network to understand the context of customer queries.

The trained Deep Learning model is integrated with a **FastAPI REST API**, while the frontend is developed using **HTML, CSS, and JavaScript**.

A user can simply enter a banking-related question such as:

> "I have been waiting for my physical card for more than ten days and it still hasn't arrived."

The system processes the text and predicts:

```text
Predicted Intent: card_arrival
Confidence: 99.4%
```

The complete system covers:

```text
Customer Query
      ↓
Text Preprocessing
      ↓
Tokenization
      ↓
Sequence Padding
      ↓
BiGRU Deep Learning Model
      ↓
77-Class Intent Prediction
      ↓
Confidence Score
      ↓
FastAPI Response
      ↓
Frontend Result
```

---

# Live Demo

Experience the deployed Banking77 Intent Prediction application:

**Live Application:**  
[Banking Issue Prediction – Live Demo](https://banking-issue-prediction-1.onrender.com/)

The deployed application allows users to enter banking-related questions and receive the predicted intent with its confidence score.

---

# Business Problem

Banking applications receive thousands of customer queries every day.

Customers may ask questions related to:

- Card delivery
- Card activation
- Cash withdrawal
- Bank transfers
- Payments
- Exchange rates
- Account verification
- Refunds
- Top-ups
- Lost or stolen cards
- Contactless payments
- Virtual cards
- Account management

Manually identifying the intent of every customer query is time-consuming and difficult to scale.

An automated intent classification system can understand the customer's message and route the query to the appropriate banking service or support workflow.

---

# Project Objective

The main objective of this project is to build a Deep Learning based NLP system that can:

- Understand natural language banking queries.
- Clean and preprocess customer text.
- Convert text into numerical sequences.
- Learn contextual relationships between words.
- Classify customer queries into 77 banking intents.
- Return the predicted intent.
- Provide a confidence score.
- Expose predictions through a REST API.
- Provide an interactive web interface.
- Deploy the complete application for real-world usage.

---

# Dataset

## Banking77

The project uses the **Banking77 dataset**, a text classification dataset containing banking-related customer queries.

The dataset contains:

```text
Training Samples : 10,003
Testing Samples  : 3,080
Number of Classes: 77
```

The dataset contains two main columns:

```text
text
label
```

Example:

```text
"I am still waiting on my card?"
        ↓
card_arrival
```

---

# Banking Intent Categories

The model is trained to classify queries into **77 different banking intents**.

Some examples include:

- activate_my_card
- age_limit
- apple_pay_or_google_pay
- atm_support
- automatic_top_up
- balance_not_updated_after_bank_transfer
- beneficiary_not_allowed
- cancel_transfer
- card_about_to_expire
- card_acceptance
- card_arrival
- card_delivery_estimate
- card_linking
- card_not_working
- card_payment_fee_charged
- card_payment_not_recognised
- card_payment_wrong_exchange_rate
- card_swallowed
- cash_withdrawal_charge
- cash_withdrawal_not_recognised
- change_pin
- compromised_card
- contactless_not_working
- country_support
- declined_card_payment
- declined_cash_withdrawal
- declined_transfer
- direct_debit_payment_not_recognised
- disposable_card_limits
- edit_personal_details
- exchange_charge
- exchange_rate
- exchange_via_app
- extra_charge_on_statement
- failed_transfer
- fiat_currency_support
- get_disposable_virtual_card
- get_physical_card
- getting_spare_card
- getting_virtual_card
- lost_or_stolen_card
- lost_or_stolen_phone
- order_physical_card
- passcode_forgotten
- pending_card_payment
- pending_cash_withdrawal
- pending_top_up
- pending_transfer
- pin_blocked
- receiving_money
- request_refund
- supported_cards_and_currencies
- terminate_account
- top_up_by_bank_transfer_charge
- top_up_by_card_charge
- top_up_by_cash_or_cheque
- top_up_failed
- transfer_not_received_by_recipient
- verify_my_identity
- why_verify_identity
- wrong_exchange_rate_for_cash_withdrawal

and other Banking77 intents.

---

# Natural Language Processing Pipeline

The NLP pipeline converts raw customer text into a format that can be understood by the Deep Learning model.

```text
Raw Customer Text
        ↓
Lowercase Conversion
        ↓
Punctuation Removal
        ↓
Tokenization
        ↓
Integer Sequences
        ↓
Padding / Truncation
        ↓
BiGRU Model
        ↓
Intent Classification
```

---

# Text Preprocessing

The project applies basic text normalization before prediction.

### Lowercase Conversion

Example:

```text
"Why Is My Card Not Working?"
```

becomes:

```text
"why is my card not working"
```

### Punctuation Removal

Punctuation is removed from the customer query before tokenization.

Example:

```text
"Why isn't my card working?"
```

becomes:

```text
"why isnt my card working"
```

---

# Tokenization

The project uses the Keras `Tokenizer` to convert words into numerical representations.

The tokenizer configuration uses:

```python
vocab_size = 2396
```

and an out-of-vocabulary token:

```text
<oov>
```

The tokenizer converts text into integer sequences that can be processed by the neural network.

---

# Sequence Padding

Different customer queries have different lengths.

To provide a fixed-size input to the model, sequences are padded/truncated to:

```text
Maximum Sequence Length = 30
```

Configuration:

```python
pad_sequences(
    sequence,
    maxlen=30,
    padding="post",
    truncating="post"
)
```

Therefore, every input is converted into a sequence of length:

```text
30
```

---

# Deep Learning Model

## Bidirectional GRU

The core Machine Learning component of this project is a **Bidirectional GRU (BiGRU)** neural network.

The architecture consists of:

```text
Input Text
    ↓
Embedding Layer
    ↓
Bidirectional GRU - 128 Units
    ↓
Dropout - 20%
    ↓
Bidirectional GRU - 64 Units
    ↓
Dropout - 20%
    ↓
Dense Layer
    ↓
Softmax
    ↓
77 Intent Classes
```

---

# Model Architecture

```python
BiGRU = Sequential([
    Embedding(
        input_dim=2396,
        output_dim=300,
        input_length=30
    ),

    Bidirectional(
        GRU(
            units=128,
            return_sequences=True
        )
    ),

    Dropout(0.2),

    Bidirectional(
        GRU(
            units=64
        )
    ),

    Dropout(0.2),

    Dense(
        units=77,
        activation="softmax"
    )
])
```

---

# Why BiGRU?

GRU stands for:

**Gated Recurrent Unit**

GRU is a type of Recurrent Neural Network designed to handle sequential data.

For NLP applications, word order and context are important.

A Bidirectional GRU processes the sequence in both directions:

```text
Forward Direction
        →
"I want to activate my new card"

Backward Direction
        ←
"I want to activate my new card"
```

This allows the model to capture contextual information from both sides of the sequence.

---

# Embedding Layer

The Embedding layer converts integer word IDs into dense numerical vectors.

Configuration:

```text
Vocabulary Size : 2396
Embedding Size  : 300
```

The embedding layer allows the network to learn meaningful numerical representations of words.

---

# GRU Layers

The architecture contains two Bidirectional GRU layers.

### First BiGRU

```text
Units = 128
Return Sequences = True
```

This layer captures sequential patterns while passing the complete sequence to the next recurrent layer.

### Second BiGRU

```text
Units = 64
```

This layer further learns higher-level contextual representations.

---

# Dropout

Two Dropout layers are used:

```text
Dropout = 0.20
```

Dropout helps reduce overfitting by randomly disabling a portion of neurons during training.

---

# Output Layer

The final Dense layer contains:

```text
77 neurons
```

with:

```text
Softmax activation
```

Each neuron represents one Banking77 intent.

The model produces a probability distribution across all 77 classes.

The class with the highest probability becomes the predicted intent.

---

# Model Training

The model was trained using:

```text
Optimizer:
Adam

Loss Function:
Sparse Categorical Crossentropy

Metric:
Accuracy

Epochs:
25

Batch Size:
32
```

The training process also uses:

- Class weights
- Early stopping
- Learning-rate reduction

These techniques help improve training stability and address class imbalance.

---

# Class Weighting

The project calculates balanced class weights using:

```python
compute_class_weight(
    class_weight="balanced"
)
```

This helps give additional importance to classes with fewer training examples.

This is particularly useful when some Banking77 intents contain fewer examples than others.

---

# Training Performance

The model progressively improves during training.

Example validation performance:

```text
Epoch 1 → 71.20%
Epoch 2 → 80.10%
Epoch 3 → 83.67%
Epoch 4 → 85.03%
Epoch 6 → 85.91%
Epoch 7 → 85.91%
```

The final evaluation on the test dataset achieved:

```text
Test Loss     : 0.5464
Test Accuracy : 86.85%
```

---

# Model Evaluation

The final model was evaluated on:

```text
3,080 Test Samples
```

Final result:

| Metric | Result |
|---|---:|
| Test Accuracy | 86.85% |
| Test Loss | 0.5464 |
| Number of Classes | 77 |
| Test Samples | 3,080 |

---

# Sample Predictions

The project tests the trained model on unseen customer queries.

### Example 1

```text
Input:
I want to activate my new card before using it.

Prediction:
activate_my_card
```

### Example 2

```text
Input:
Why is my contactless payment not working at stores?

Prediction:
contactless_not_working
```

### Example 3

```text
Input:
My identity verification keeps getting rejected.

Prediction:
unable_to_verify_identity
```

### Example 4

```text
Input:
The exchange rate used for my cash withdrawal is incorrect.

Prediction:
wrong_exchange_rate_for_cash_withdrawal
```

### Example 5

```text
Input:
I cannot link my existing card to the app.

Prediction:
card_linking
```

---

# Prediction Confidence

The model does not only return the predicted intent.

It also calculates the confidence of the prediction.

```python
prediction = model.predict(padded)

idx = np.argmax(prediction)

confidence = np.max(prediction)
```

The API returns:

```json
{
    "predicted_intent": "card_arrival",
    "confidence": 0.994
}
```

This allows the application to communicate how confident the model is about its classification.

---

# FastAPI Backend

The trained BiGRU model is integrated into a **FastAPI REST API**.

FastAPI provides the bridge between the Deep Learning model and the web frontend.

Backend technologies:

```text
FastAPI
TensorFlow / Keras
Pydantic
NumPy
Pickle
```

---

# API Architecture

```text
                User
                 │
                 ▼
        HTML / CSS / JavaScript
                 │
                 │ HTTP POST
                 ▼
              FastAPI
                 │
                 ▼
          Pydantic Schema
                 │
                 ▼
        Text Preprocessing
                 │
                 ▼
             Tokenizer
                 │
                 ▼
          Sequence Padding
                 │
                 ▼
             BiGRU Model
                 │
                 ▼
        77-Class Prediction
                 │
                 ▼
        Confidence Calculation
                 │
                 ▼
             JSON Response
                 │
                 ▼
             Frontend
```

---

# API Endpoints

## Home Endpoint

```http
GET /
```

Returns the API status.

Example:

```json
{
    "message": "Banking77 Intent Prediction API is running successfully!"
}
```

---

## Prediction Endpoint

```http
POST /predict
```

This endpoint accepts a customer query.

### Request

```json
{
    "text": "I have been waiting for my card for more than ten days"
}
```

### Response

```json
{
    "input_text": "I have been waiting for my card for more than ten days",
    "predicted_intent": "card_arrival",
    "confidence": 0.994
}
```

---

# Pydantic Validation

The API uses Pydantic to define the request structure.

```python
class UserQuery(BaseModel):
    text: str
```

This ensures the API receives the expected input format.

---

# Frontend

The frontend is developed using:

- HTML
- CSS
- JavaScript

The interface provides a simple banking-support style experience where users can type their banking query and classify it.

---

# Frontend Features

- Clean banking-focused UI
- Customer query input
- Classify button
- Example query suggestions
- Predicted intent display
- Confidence percentage
- Query preview
- Responsive interface
- API integration
- Real-time prediction response

---

# Application Flow

```text
1. User enters banking question
             ↓
2. JavaScript sends request
             ↓
3. FastAPI receives request
             ↓
4. Pydantic validates input
             ↓
5. Text is cleaned
             ↓
6. Tokenizer converts text to sequence
             ↓
7. Sequence is padded to 30 tokens
             ↓
8. BiGRU predicts 77-class probability
             ↓
9. Highest probability class selected
             ↓
10. Confidence calculated
             ↓
11. JSON response returned
             ↓
12. Frontend displays prediction
```

---

# Model Files

The project uses three important serialized files.

### BiGRU Model

```text
BiGRU_model.keras
```

Contains the trained Deep Learning model.

### Tokenizer

```text
tokenizer.pkl
```

Stores the vocabulary and word-to-index mapping used during training.

### Label Names

```text
label_names.pkl
```

Stores the mapping between output class indexes and Banking77 intent names.

Together, these files allow the trained model to be reused for inference.

---

# Project Structure

```text
Banking_issue_prediction/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── static/
│   ├── BiGRU_model.keras
│   ├── tokenizer.pkl
│   └── label_names.pkl
│
├── Banking77.jpeg
├── Banking_text_Prediction.ipynb
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| NLP | Keras Tokenizer |
| Neural Network | BiGRU |
| Data Processing | NumPy, Pandas |
| Backend | FastAPI |
| Validation | Pydantic |
| Frontend | HTML |
| Styling | CSS |
| Client-side Logic | JavaScript |
| Model Format | Keras |
| Serialization | Pickle |
| Deployment | Render |

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Banking_issue_prediction.git
```

## 2. Navigate to the Project

```bash
cd Banking_issue_prediction
```

## 3. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

---

# Open the Frontend

The FastAPI application serves the frontend through:

```text
/app
```

Open:

```text
http://127.0.0.1:8000/app
```

---

# API Documentation

FastAPI automatically provides interactive API documentation.

After starting the server, open:

```text
http://127.0.0.1:8000/docs
```

You can use Swagger UI to test:

```http
POST /predict
```

---

# Deployment

The application is deployed using **Render**.

## Live Application

[Open Banking77 Intent Prediction Application](https://banking-issue-prediction-1.onrender.com/)

The deployed application can be used directly without installing the project locally.

---

# Real-World Applications

This type of intent classification system can be used for:

- Banking Customer Support
- Chatbots
- Virtual Banking Assistants
- Automated Ticket Classification
- Customer Query Routing
- FAQ Automation
- Support Ticket Prioritization
- Conversational AI
- Banking Helpdesk Automation

---

# Business Impact

An automated intent classification system can help banking platforms:

- Reduce manual query classification.
- Route customer requests to appropriate departments.
- Improve support automation.
- Build intelligent banking chatbots.
- Improve customer response time.
- Reduce repetitive support tasks.
- Create scalable customer-service workflows.

---

# Key Challenges

During development, several NLP and Deep Learning challenges were addressed.

### 1. Multiple Intent Categories

The model must distinguish between 77 different banking intents.

Some intents are semantically similar, making classification challenging.

For example:

```text
card_arrival
card_delivery_estimate
order_physical_card
```

The model needs to learn the contextual differences between these queries.

### 2. Variable Text Length

Customer queries have different lengths.

This was handled using:

```text
Tokenization
+
Padding
+
Truncation
```

### 3. Class Imbalance

Class weights were used to reduce the impact of uneven class distributions.

### 4. Context Understanding

BiGRU was selected to capture contextual information from both directions of a sequence.

---

# Skills Demonstrated

This project demonstrates practical knowledge of:

## Deep Learning

- Neural Networks
- Recurrent Neural Networks
- GRU
- Bidirectional GRU
- Embeddings
- Dropout
- Softmax
- Model Training
- Early Stopping
- Learning Rate Scheduling

## Natural Language Processing

- Text Cleaning
- Tokenization
- Vocabulary Creation
- Sequence Encoding
- Padding
- Truncation
- Intent Classification

## Machine Learning Engineering

- Model Serialization
- Inference Pipeline
- Model Integration
- API Deployment
- Production-style Architecture

## Backend Development

- FastAPI
- REST APIs
- Pydantic
- CORS
- API Routing
- JSON Responses

## Frontend Development

- HTML
- CSS
- JavaScript
- REST API Integration
- Interactive UI
- Responsive Design

---

# Project Highlights

```text
Banking77 Dataset
       ↓
77 Banking Intents
       ↓
Text Preprocessing
       ↓
Keras Tokenizer
       ↓
Sequence Padding
       ↓
Embedding Layer
       ↓
Bidirectional GRU
       ↓
77-Class Softmax
       ↓
86.85% Test Accuracy
       ↓
FastAPI REST API
       ↓
HTML + CSS + JavaScript
       ↓
Render Deployment
```

---

# Future Improvements

The project can be further enhanced with:

- Transformer-based NLP models
- BERT
- DistilBERT
- RoBERTa
- Transfer Learning
- Confidence Thresholding
- Low-confidence fallback handling
- Conversation history
- Multi-turn banking chatbot
- Intent-based FAQ retrieval
- RAG integration
- Authentication
- Database integration
- Docker deployment
- CI/CD pipeline
- Model monitoring
- MLflow experiment tracking
- Explainable AI
- Automated model retraining

---

# Conclusion

The **Banking77 Intent Prediction** project demonstrates a complete Deep Learning and NLP workflow, starting from raw customer text and ending with a deployed web application.

The system uses a **Bidirectional GRU model** to classify banking queries into **77 different intents**, achieving approximately **86.85% test accuracy**.

By combining:

```text
NLP
+
Deep Learning
+
BiGRU
+
FastAPI
+
HTML
+
CSS
+
JavaScript
+
Cloud Deployment
```

the project demonstrates how a Deep Learning model can be transformed into a practical end-to-end application.

---

# Author

## Parth Solanki

**Machine Learning Engineer | AI/ML Enthusiast **

---

## License

This project is developed for educational, learning, and portfolio purposes.
