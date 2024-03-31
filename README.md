# Comparison of Transfer Learning Techniques for Building Energy Forecasting

<i>Das Sharma, S., Coursey, A., Quinones-Grueiro, M., & Biswas, G.</br>
Department of Computer Science, Vanderbilt University, TN, USA</br>
12th IFAC Symposium on Fault Detection, Supervision and Safety for Technical Processes</i>

### **:pencil2: Abstract**:

The growing demand for building energy efficiency necessitates accurate predictions of normal versus abnormal operations to understand their impact on energy management. However, integrating predictive models into practical applications faces challenges, especially in buildings with limited measurements and data. This paper explores the viability of three widely adopted transfer learning techniques in improving energy consumption models, focusing on real-world data with internal building measurements. The findings suggest that transferring information between buildings is a promising method to provide positive improvements in energy prediction models.

### **:computer: Setup**:

To install required packages, run:

```
$ pip install -r requirements.txt
```

Run through `experiments/model_comparison.ipynb` to train and test all models to populate `transfer_results.json` (note that this is a slow process). Then run through `experiments/result_displays.ipynb` to generate comparative plots. Methods 1, 2, and 3 from the paper are benchmarked in this manner.

R-DANN methodology was run separately and included in this repository under the `dann-transfer` subfolder.

### **:open_file_folder: Directory Structure**:

```
.
├── README.md
│
├── base_models
│ ├── model_base_{autoLSTM, LD, MLP}.py: : Model architecture and generation functions
│ ├── create_base_{autoLSTM, LD, MLP}.ipynb: Jupyter Notebooks to call the model generation functions
│
├── experiments
│ ├── model_comparison.ipynb: Calls relevant methods in the repo to generate all models and populates transfer_results.json with relevant details for benchmarking
│ └── result_displays.ipynb: Utilizes populated transfer_results.json to generate visualizations
│
├── helper
│ ├── grid_search.ipynb
│ └── mlp_grid_search.ipynb
│
├── preprocessing
│ ├── esb_preprocessing.ipynb
│ ├── kissam_preprocessing.ipynb
│ └── preprocessor.py
│
├── requirements.txt
│
├── results
│ └── result_data
│ ├── alt_transfer_results.json
│ ├── data_amounts.csv
│ └── transfer_results.json
│
└── transfer_logic
├── adjustedtransfer_runthrough.ipynb
├── transfer_LD_weightinit.py
├── transfer_adjusted.py
└── transfer_autoLSTM.py
```
