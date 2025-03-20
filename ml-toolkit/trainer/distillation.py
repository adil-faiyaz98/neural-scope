from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

class Distillation:
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 optimizer: optim.Optimizer, criterion: nn.Module, 
                 alpha: float = 0.5, temperature: float = 2.0):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.alpha = alpha
        self.temperature = temperature

    def train(self, train_loader: Any, num_epochs: int) -> None:
        self.teacher_model.eval()
        self.student_model.train()

        for epoch in range(num_epochs):
            for data, target in train_loader:
                self.optimizer.zero_grad()

                # Teacher predictions
                with torch.no_grad():
                    teacher_output = self.teacher_model(data)

                # Student predictions
                student_output = self.student_model(data)

                # Calculate loss
                loss = self._distillation_loss(student_output, teacher_output, target)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    def _distillation_loss(self, student_output: torch.Tensor, 
                           teacher_output: torch.Tensor, 
                           target: torch.Tensor) -> torch.Tensor:
        # Soft targets
        soft_target_loss = nn.KLDivLoss()(nn.functional.log_softmax(student_output / self.temperature, dim=1),
                                           nn.functional.softmax(teacher_output / self.temperature, dim=1)) * (self.temperature ** 2)

        # Hard targets
        hard_target_loss = self.criterion(student_output, target)

        return self.alpha * soft_target_loss + (1 - self.alpha) * hard_target_loss

    def evaluate(self, test_loader: Any) -> Tuple[float, float]:
        self.student_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = self.student_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        return accuracy, total