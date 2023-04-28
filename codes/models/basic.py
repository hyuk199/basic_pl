import torch
import pytorch_lightning as pl

from typing import Dict, List, Tuple

class BasicModel(pl.LightningModule):
    def __init__(self,
                 lr = 1.0e-5,
                 **kwargs):
        super().__init__()
        self.modelname = "basic"

        """ Basic model
            Need to set 
                __init__()
                forward() : 
                compute_metrics() : metrics like loss and accuracy 
                configure_optimizers()
                output_dico() : validation & test outputs(metrics or prediction) to dictionary for log_dict
            
            _shared_eval_step() : validation & test step
        """
        super().__init__()

        self.lr = lr
        self.valid_step_outputs = []
        self.test_step_outputs = []

    def forward(self, inputs):
        pass

    def compute_metrics(self, outputs, targets)-> Tuple[Dict, torch.Tensor] :
        metrics = {}
        loss = metrics['']
        return metrics, loss

    def training_step(self, batch, batch_idx):
        output = self.forward(batch["inputs"])
        metrics, loss = self.compute_metrics(output, batch["targets"])
        self.log_dict(metrics)
        return loss

    # ----- Eval Steps -----
    def _shared_eval_step(self, batch, batch_idx):
        output = self.forward(batch["inputs"])
        metrics, loss = self.compute_metrics(output, batch["targets"])
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        self.valid_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        self.log_dict(self.output_dico(self.valid_step_outputs))
        self.valid_step_outputs.clear()
        
    def test_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        self.log_dict(self.output_dico(self.test_step_outputs))
        self.test_step_outputs.clear()

    def output_dico(self, outputs:List)-> Dict:
        dico = {}
        return dico
    # ----- Optimizer Steps -----
    def configure_optimizers(self):
        pass
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}