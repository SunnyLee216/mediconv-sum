import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer

class PPOTextSummarization(pl.LightningModule):
    def __init__(self, lr, gamma, clip_eps):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.model = AutoModel.from_pretrained("t5-base")
        
    def forward(self, x):
        input_ids = self.tokenizer.encode(x, return_tensors="pt")
        output_ids = self.model.generate(input_ids)
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y_hat = self(x)
        
        rouge = calculate_rouge(y_hat, y)
        reward = torch.tensor(rouge, dtype=torch.float32)
        
        old_policy = self(x)
        old_probs = torch.softmax(old_policy, dim=-1)
        old_log_probs = torch.log_softmax(old_policy, dim=-1)
        old_value = self.critic(self.model(x)[0][:, 0, :])
        
        for i in range(self.num_steps):
            policy = self(x)
            probs = torch.softmax(policy, dim=-1)
            log_probs = torch.log_softmax(policy, dim=-1)
            
            kl_div = kl_divergence(probs, old_probs)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            value = self.critic(self.model(x)[0][:, 0, :])
            
            advantage = reward - old_value
            surr_loss = -torch.min(
                advantage.detach() * log_probs - self.clip_eps * kl_div, 
                advantage.detach() * old_log_probs - self.clip_eps * kl_div
            )
            policy_loss = torch.mean(surr_loss - self.beta * entropy)
            
            value_loss = torch.nn.functional.mse_loss(value, reward)
            
            loss = policy_loss + value_loss
            
            self.log("loss", loss)
            self.log("policy_loss", policy_loss)
            self.log("value_loss", value_loss)
            self.log("reward", reward)
            self.log("rouge", rouge)
            self.log("kl_div", kl_div)
            self.log("entropy", entropy)
            self.log("advantage", advantage)
            self.log("value", value)
            
            old_policy = policy.detach()
            old_probs = probs.detach()
            old_log_probs = log_probs.detach()
            old_value = value.detach()
            
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        rouge = calculate_rouge(y_hat, y)
        reward = torch.tensor(rouge, dtype=torch.float32)
        
        return {"val_reward": reward}
    def validation_epoch_end(self, outputs):
        avg_reward = torch.stack([x["val_reward"] for x in outputs]).mean()
        
        self.log("val_reward", avg_reward, prog_bar=True)
        return {"val_reward": avg_reward}
     def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        rouge = calculate_rouge(y_hat, y)
        self.log("test_rouge", rouge)
    def calculate_rouge(pred,target):
        pass
    def kl_divergence(prods,old_prods):
        pass
