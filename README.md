# Improving Customer Experience with Predictive Analytics for Replenishment Recommendations

## Project Descirption
This project focuses on developing a robust predictive analytics system that anticipates which products a customer is likely to repurchase and predicts the timing of those purchases. Leveraging advanced machine learning algorithms—including XGBoost, Random Forest and LSTM our solution learns patterns from historical transaction data and user behavior, providing personalized, time-aware product recommendations. By predicting repurchases for everyday household items, the system aims to not only enhance customer experience through timely and relevant suggestions but also optimize inventory management for retailers by ensuring the right products are in stock when customers need them. Key metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and precision guide model selection and fine-tuning, ensuring robust performance. Ultimately, the ability to provide proactive reordering recommendations significantly improves the consumer’s shopping journey while supporting operational efficiency across the supply chain.

## Team Members
- Hasan Taha Bağcı - 150210338
- Selman Turan Toker - 150220330

## Dataset


## Project Structure
```bash
│
├── README.md
│
├── data
│   ├── product_catalog.csv
│   ├── product_category_map.csv
│   └── test.csv
│   └── transactions.csv
│
├── docs
│   ├── Project_Proposal.pdf
│   └── Final_Report.pdf (will be added)
│
├── utils
│   ├── data_preprocess.py
│   ├── model_process.py
│   └── plots.py
│    
├── models
│   ├── rf_model_custom.pkl
│   ├── xgb_model_custom.pkl
│   └── pytorch_lstm_model.pth
│
├── main.ipynb
│
└── requirements.txt
```

## Installation
Required packages:
- Python3
- jupyter
- joblib==1.3.2
- matplotlib==3.10.0
- numpy==2.2.1
- pandas==2.2.3
- scikit_learn==1.6.0
- seaborn==0.13.2
- xgboost==2.1.3

To install required packages run `pip install -r requirements.txt` in the project directory.

## Usage
To run the project, run `jupyter notebook` in the project directory and open `main.ipynb` file.
You can run the cells in the notebook to see the data exploration process and create the `final_data.csv` file. 
After that, you can run the cells in the model notebook to see the model training process and the results, anlysis and evaluation of the models.