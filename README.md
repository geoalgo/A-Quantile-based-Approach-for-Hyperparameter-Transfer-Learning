# A-Quantile-based-Approach-for-Hyperparameter-Transfer-Learning

This code reproduces the method of the paper 
[A quantile approach for hyperparameter transfer learning](https://icml.cc/virtual/2020/poster/6483) published at 
ICML 2020. 

Four datasets of precomputed evaluations are available (DeepAR, FCNET, XGBoost and nas_bench102). The first three 
 are taken from this [repo](https://github.com/icdishb/hyperparameter-transfer-learning-evaluations), the last 
 one was generated from [NAS-bench-102](https://github.com/Debrove/NAS-Projects)).

This is an reimplementation from scratch of the method as the initial implemented depended on proprietary components. 
As such, the results differ and may be slightly worse given that different frameworks were used (botorch,
 pytorch for the GP in particular) but we made sure the reimplementation is reasonably close, see below for a 
 comparison with the reimplementation and our initial implementation.
 
*Table 2 comparison*

The two reimplementations are indicated with \*, two parameteric priors have been implemented with sklearn and pytorch.

```
blackbox               DeepAR     FCNET   XGBoost  nas_bench102
optimizer                                                      
RS                   0.000000  0.000000  0.000000      0.000000
CTS                  0.378205  0.825543  0.015180      0.876395
CTS* (sklearn)       0.541521  0.790564  0.170170      0.859444
CTS* (pytorch)       0.567947  0.763614  0.140947      0.956671
GCP + prior          0.733711  0.942026  0.371962      0.940347
GCP+prior* (sklearn) 0.933025  0.780496 -0.012777      0.969306
GCP+prior* (pytorch) 0.389110  0.769105  0.137245      0.978004
```

I have cleaned the code so that Figure 1, Figure 2 and Table 2 can be recomputed in case you want to compare with your 
new method or evaluate again the proposed method.  

## How to run

### setup interpreter

I recommend using the conda environment used for the evaluations is given in `environment-GC3P.yml`.
```
conda env create -f environment-GC3P.yml
conda activate GC3P
```

The requirements.txt is done manually and may give you something similar with less guarantees.

### figure and table generation

Regenerate figures and tables.
```
python experiments/figure1.py
python experiments/figure2.py
python experiments/table2.py
python experiments/figure_illustration.py

# plot the value indicated above
python experiments/table2-new-implem.py
```

Those script will use precomputed-results under `results_paper.csv.zip` and `results_reimplementation.csv.zip`.

The format such files is:

```
task,optimizer,seed,iteration,value,blackbox
german.numer,GCP + prior (ours),21,1,0.25602100000000005,XGBoost
...
```

### rerun new implementation

To rerun the reimplementation, you can use the following command to obtain results on a given method/task: 
```  
python experiments/evaluate_optimizer_task.py --task cifar10 --optimizer GCP+prior --num_seeds 2 --num_evaluations 20 --output_folder result-folder/
```

This will write `result.csv.zip` into the folder specified in output_folder.
As the result will be for a single task/optimizer, to run all the benchmark we highly recommend parralelizing through
your favorite cloud provider.


### benchmark 

To benchmark some methods on a given dataset quickly, see `benchmark_example.py`. This will run the optimizers
and plot their convergence distribution.


## Citation

In case this work is useful for your research, here is a bibtex for you know what :)

```
@incollection{icml2020_4367,
 author = {Salinas, David and Shen, Huibin and Perrone, Valerio},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {7706--7716},
 title = {A quantile-based approach for hyperparameter transfer learning},
 year = {2020}
}
```


In case you have any question, please feel free to open an issue or send me an email.

