"""Predict and inject a short prefix inside one native draft transformer pass."""
import torch


def predicted_embedding(logits, embedding, straight_through=False, temperature=1.):
    hard = embedding(logits.argmax(-1))
    if not straight_through:
        return hard
    probabilities = (logits.float()/temperature).softmax(-1).to(embedding.weight.dtype)
    soft = probabilities @ embedding.weight
    return hard+(soft-soft.detach())


class MidpointConditioning:
    def __init__(self, student, target, config):
        self.student, self.target = student, target
        self.prefix = config["prefix"]
        self.strength = config["strength"]
        self.temperature = config.get("temperature", 1.)
        self.capture = False
        self.logits = None
        depth = config["after_layers"]
        if not 1 <= depth < len(student.layers) or not 0 < self.prefix < student.block_size-1:
            raise ValueError("Midpoint must precede remaining draft layers and leave a tail")
        if self.temperature <= 0 or not 0 <= self.strength <= 1:
            raise ValueError("Invalid midpoint temperature or strength")
        self.handle = student.layers[depth-1].register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        if not self.capture and self.strength == 0:
            return output
        if output.shape[1] <= self.prefix:
            raise RuntimeError("Draft block is too short for midpoint prefix")
        logits = self.target.lm_head(self.student.norm(output[:, 1:self.prefix+1]))
        self.logits = logits if self.capture else None
        if self.strength == 0:
            return output
        hints = predicted_embedding(logits, self.target.model.embed_tokens, torch.is_grad_enabled(), self.temperature)
        mask = self.target.model.embed_tokens.weight[self.student.mask_token_id]
        changed = output.clone()
        changed[:, 1:self.prefix+1] = output[:, 1:self.prefix+1]+self.strength*(hints-mask)
        return changed

    def take_logits(self):
        logits, self.logits = self.logits, None
        if logits is None:
            raise RuntimeError("Missing intermediate prediction supervision")
        return logits
