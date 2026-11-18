"""
Wald SPRT on one model's per-trial correctness (Bernoulli), used as the online finetune gate.
  H0: accuracy = p0 (bad)  -> score <= lower = ln(beta / (1 - alpha))  -> "finetune"
  H1: accuracy = p1 (good) -> score >= upper = ln((1 - beta) / alpha)  -> "good"
Correct trial: + ln(p1 / p0); wrong trial: + ln((1 - p1) / (1 - p0)); unscored trial (None): no step.
The score goes back to 0 after every decision.
"""
import math


class SprtGate:
    def __init__(self, p0=0.5, p1=0.75, alpha=0.05, beta=0.2):
        if not (0 < p0 < p1 < 1):
            raise ValueError(f"need 0 < p0 < p1 < 1, got p0={p0}, p1={p1}")
        if not (0 < alpha < 1 and 0 < beta < 1 and alpha + beta < 1):
            raise ValueError(f"need 0 < alpha, beta and alpha + beta < 1, got alpha={alpha}, beta={beta}")
        self.p0, self.p1, self.alpha, self.beta = p0, p1, alpha, beta
        self.win = math.log(p1 / p0)
        self.loss = math.log((1 - p1) / (1 - p0))
        self.upper = math.log((1 - beta) / alpha)
        self.lower = math.log(beta / (1 - alpha))
        self.score = 0.0
        self.last_score = 0.0  # score right after the latest step, before any reset (for logging)

    def step(self, correct):
        if correct is None:
            return 0.0
        return self.win if correct else self.loss

    def update(self, correct):
        """correct: True / False / None. Returns "finetune", "good" or None."""
        self.score += self.step(correct)
        self.last_score = self.score
        if correct is None:
            return None
        if self.score <= self.lower:
            self.reset()
            return "finetune"
        if self.score >= self.upper:
            self.reset()
            return "good"
        return None

    def reset(self):
        self.score = 0.0
