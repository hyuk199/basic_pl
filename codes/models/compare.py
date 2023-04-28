import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Dict, List, Tuple

class CompareModel(pl.LightningModule):
    def __init__(self,
                 base_model,
                 lr = 1.0e-5,
                 **kwargs):
        super().__init__()
        self.modelname = "compare"

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
        self.base_model = base_model
        self.v_head = nn.Linear(base_model.n_embd, 1, bias=False)

        self.valid_step_outputs = []
        self.test_step_outputs = []

    def forward(self, inputs):
        loss = None

        base_model_outputs = self.base_model(inputs)
        hidden_states = base_model_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        self.PAD_ID

        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(inputs.shape) == 2
        bs = inputs.shape[0] // 2
        chosen = inputs[:bs]
        rejected = inputs[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the last value before padding
        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }

    def compute_metrics(self, eval_preds):
        chosen_end_scores = eval_preds["chosen_end_scores"]  # chosen scores
        rejected_end_scores = eval_preds["rejected_end_scores"]  # rejected scores
        loss = eval_preds["loss"]

        acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
        
        return acc, loss

    def training_step(self, batch, batch_idx):
        dico = {}
        output = self.forward(batch["inputs"])
        loss = output["loss"]

        dico.update({"train_loss": loss})
        self.log_dict(dico)
        return loss

    # ----- Eval Steps -----
    def _shared_eval_step(self, batch, batch_idx):
        output = self.forward(batch["inputs"])
        metrics, loss = self.compute_metrics(output, batch["targets"])
        return {"loss": loss, "acc": metrics}

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
        loss = [output['loss'] for output in outputs]
        _loss = sum(loss)/len(loss)
        acc = [output['acc'] for output in outputs]
        _acc = sum(acc)/len(acc)
                   
        dico.update({"loss": _loss, "acc": _acc})
        return dico
    # ----- Optimizer Steps -----
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer