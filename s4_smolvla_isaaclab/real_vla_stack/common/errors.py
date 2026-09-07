class ContractError(ValueError):
    """A dataset, checkpoint or wire payload violates the deployed contract."""


class DataValidationError(ValueError):
    """Raw or converted data is unsafe to consume."""


class PolicyStaleError(RuntimeError):
    """No fresh policy trajectory is available for robot execution."""


class RobotStateStaleError(RuntimeError):
    """Robot feedback is too old to support closed-loop motion."""


class CommandRouteConflictError(RuntimeError):
    """Another publisher has appeared on the commissioned command route."""


class CommandOutputRelinquishedError(RuntimeError):
    """The hardware watchdog has already relinquished command output."""


class RobotTrackingError(RuntimeError):
    """The measured arm is not following the last published target."""
