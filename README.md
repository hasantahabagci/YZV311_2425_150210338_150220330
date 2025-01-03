# Enhancing Customer Experience with AI-Driven Purchase Forecasting

## Project Descirption
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam nec pur

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
├── main.ipynb
│
└── requirements.txt
```

## Installation
Required packages:
- Python3
- jupyter
- pandas==1.4.4
- scikit-learn==1.0.2
- seaborn==0.11.2
- xgboost==1.7.5
- matplotlib==3.5.2

To install required packages run `pip install -r requirements.txt` in the project directory.

## Usage
To run the project, run `jupyter notebook` in the project directory and open `exploration.ipynb` and `model.ipynb` files.
You can run the cells in the exploration notebook to see the data exploration process and create the `processed_data.csv` file. 
After that, you can run the cells in the model notebook to see the model training process and the results, anlysis and evaluation of the models.