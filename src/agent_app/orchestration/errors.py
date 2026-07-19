"""Прикладные исключения для распределённой оркестрации."""


class OrchestrationError(RuntimeError):
    """Базовая ошибка orchestration runtime."""


class TransientOrchestrationError(OrchestrationError):
    """Временная ошибка, для которой разрешён retry очереди."""


class QueueCapacityError(OrchestrationError):
    """Backpressure отклонил новое задание из-за заполненной очереди."""


class JobNotFoundError(OrchestrationError):
    """Задание с указанным идентификатором не найдено."""
