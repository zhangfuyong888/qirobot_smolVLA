class ContractError(ValueError):
    """A dataset, checkpoint or wire payload violates the deployed contract."""


class DataValidationError(ValueError):
    """Raw or converted data is unsafe to consume."""


class PolicyStaleError(RuntimeError):
    """No fresh policy trajectory is available for robot execution."""
