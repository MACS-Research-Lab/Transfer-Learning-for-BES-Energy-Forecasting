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

1. Generate base models by running through create*base*{autoLSTM, LD, MLP}.ipynb for the cases you wish to observe
2. Following the logic in [`experiments/model_comparison.ipynb`](./experiments/model_comparison.ipynb), run through the code for the specific case to generate transfer results that populate [`transfer_results.json`](./results/result_data/transfer_results.json)
3. To train and test all cases of transfers between building, run through [`experiments/model_comparison.ipynb`](./experiments/model_comparison.ipynb) to populate `transfer_results.json` (note that this is a slow process). Then run through [`transfer_results.json`](./results/result_data/transfer_results.json) to generate comparative plots.

Methods 1, 2, and 3 from the paper are benchmarked in this manner. R-DANN methodology was run separately and included in this repository under the [`r-dann`](./r-dann/) subfolder.

### **:open_file_folder: Directory Structure**:

```
.
├── README.md
├── base_models
│ ├── model_base_{autoLSTM, LD, MLP}.py: : Model architecture and generation functions
│ ├── create_base_{autoLSTM, LD, MLP}.ipynb: Jupyter Notebooks to call the model generation functions
├── data
│   ├── esb : 1-year data for the Engineering Science Building
│   │   ├── 2422_1.csv : Raw data
│   │   ├── 2621_1.csv : Raw data
│   │   ├── 2841_2.csv : Raw data
│   │   ├── esb1_preprocessed.csv : Preprocessed data generated after using functions in preprocessing/
│   │   └── esb2_preprocessed.csv : Preprocessed data generated after using functions in preprocessing/
│   └── kissam : 1-year data for the Kissam Building
│       ├── 1482_1.csv : Raw data
│       ├── 1509_2.csv : Raw data
│       ├── 1510_x.csv : Raw data
│       ├── 2661_x.csv : Raw data
│       ├── 4684_1.csv : Raw data
│       ├── 4685_2.csv : Raw data
│       ├── kissam1_preprocessed.csv: Preprocessed data generated after using functions in preprocessing/
│       └── kissam2_preprocessed.csv: Preprocessed data generated after using functions in preprocessing/
├── experiments
│   ├── model_comparison.ipynb : Calls relevant methods in the repo to generate all models and populates transfer_results.json with relevant details for benchmarking
│   └── result_displays.ipynb : Utilizes populated transfer_results.json to generate visualizations
├── helper
│   ├── grid_search.ipynb : Latest grid search code used to optimize LSTM-backed models (was modified continually while optimizing each paramatere one-by-one)
│   └── mlp_grid_search.ipynb : Latest grid search code used to optimize MLP model
├── preprocessing
│   ├── esb_preprocessing.ipynb : Notebook to run through ESB building preprocessing (performed one-by-one for each tower)
│   ├── kissam_preprocessing.ipynb : Notebook to run through Kissam preprocessing (performed one-by-one for each tower)
│   └── preprocessor.py
├── r-dann : Subdirectory for R-DANN logic. Details within the sub-README
│   ├── README.md
│   ├── data_loader.py
│   ├── dataset
│   │   └── esb
│   │       ├── esb1_preprocessed.csv
│   │       └── esb2_preprocessed.csv
│   ├── functions.py
│   ├── main.py
│   ├── model.py
│   ├── model_prep.py
│   └── test.py
├── requirements.txt
├── results : Results and models generated are to be saved here
│   └── result_data
│       ├── data_amounts.csv
│       └── transfer_results.json
└── transfer_logic : Code to implement each of the 3 techniques of transfer learning
    ├── transfer_LD_weightinit.py
    ├── transfer_adjusted.py
    └── transfer_autoLSTM.py
```
