# basic_pl
basic code for [pytorch lightning 2.0](https://www.pytorchlightning.ai/index.html) and [hydra](https://github.com/facebookresearch/hydra)

There is basic of ```model, dataset, utils``` python code to Modify for your own purposes.

To run train.py you need to check ```configs/train.yaml```

The simplest way to make sure that you have all dependencies in place is to use
[conda](https://docs.conda.io/projects/conda/en/4.6.1/index.html). You can
create a conda environment called ```basic_pl``` using. 

You can use any name if you want.
```
conda create -n basic_pl python=3.9
conda activate basic_pl

pip install pytorch_lightning
pip install hydra-core hydra_colorlog rich 
pip install pandas transformers psutil

```

This is for my own personal use, but people are welcome to use it if they need it.
