import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface
from scipy import sparse


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        value = self.lambda_ * pow((self.s0/(self.s0 + iteration)), self.p)
        return value
        # TODO: реализовать формулу затухающего шага обучения
        raise NotImplementedError()


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        self.iteration = 0
        loss_history = [self.model.compute_loss()]
        for _ in range(self.max_iter):
            delta = self._step()
            loss = self.model.compute_loss()
            loss_history.append(loss)

            if np.isnan(delta).any():
                break
            if delta @ delta < self.tolerance:
                break
        self.model.loss_history = loss_history
        # TODO: implement
        # в конце также приcваивает атрибуту модели полученный loss_history


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать vanilla градиентный спуск
        # Можно использовать атрибуты класса self.model
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients()
        delta = - self.lr_schedule.get_lr(self.iteration) * gradient
        self.model.w += delta

        return delta
        raise NotImplementedError()


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели

        indexes = np.random.randint(len(self.model.y_train), size = self.batch_size)
        gradient = self.model.compute_gradients(self.model.X_train[indexes], self.model.y_train[indexes])
        delta = - self.lr_schedule.get_lr(self.iteration) * gradient
        self.model.w += delta

        return delta
        raise NotImplementedError()


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать SAG
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.avg_grad = np.zeros(num_features)
            # TODO: инициализировать хранилища при первом вызове 

        indexes = np.random.choice(num_objects, size=min(self.batch_size, num_objects), replace=False)
        X_batch = X_train[indexes]
        if sparse.issparse(X_batch):
            X_batch = X_batch.toarray()
        new = self.model.compute_individual_gradients(X_batch, y_train[indexes])
        old = self.grad_memory[indexes]
        self.avg_grad += np.sum(new - old, axis=0) / num_objects
        self.grad_memory[indexes] = new
        learning_rate = self.lr_schedule.get_lr(self.iteration)
        delta = - learning_rate * self.avg_grad
        self.model.w += delta
        return delta
    
        # TODO: реализовать SAG
        raise NotImplementedError()


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.iteration == 0:
            self.velocity = np.zeros_like(self.model.w)
        g = self.model.compute_gradients()
        self.velocity = self.beta * self.velocity + self.lr_schedule.get_lr(self.iteration) * g
        delta = - self.velocity.copy()
        self.model.w += delta
        return delta
        # TODO: реализовать градиентный спуск с моментумом
        raise NotImplementedError()


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        if self.iteration == 0:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)
        g = self.model.compute_gradients()
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
        t = self.iteration + 1
        m_hat = self.m / (1 - pow(self.beta1, t))
        v_hat = self.v / (1 - pow(self.beta2, t))
        delta = - self.lr_schedule.get_lr(self.iteration) * m_hat / (np.sqrt(v_hat) + self.eps)
        self.model.w += delta
        
        return delta
        # TODO: реализовать Adam по формуле из ноутбука
        raise NotImplementedError()


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        model = self.model
        loss = model.loss_function
        X, y = model.X_train, model.y_train
        w = loss.analytic_solution(X, y)
        model.w = w
