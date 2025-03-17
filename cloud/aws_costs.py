"""
Module: aws_costs.py

Provides AWS cost insight functionality.
Includes static data for AWS instance performance and cost, and functions to estimate cost for given computational demand.
"""
import logging

logger = logging.getLogger(__name__)

# Static data for some AWS instance types: cost (USD/hour) and estimated operations per second capacity
INSTANCE_COSTS = {
    "t2.micro": {"cost_per_hour": 0.0116, "ops_per_sec": 100000},  # small instance, slower
    "m5.large": {"cost_per_hour": 0.096, "ops_per_sec": 500000},  # general purpose instance
    "c5.large": {"cost_per_hour": 0.085, "ops_per_sec": 1000000},  # compute optimized instance
    "p3.2xlarge": {"cost_per_hour": 3.06, "ops_per_sec": 10000000},  # GPU instance (rough estimate)
}


def estimate_cost(operations: int, instance_type: str = "m5.large") -> float:
    """
    Estimate the AWS cost in USD to perform the given number of operations on the specified instance type.

    :param operations: Number of atomic operations.
    :param instance_type: AWS instance type to estimate cost for.
    :return: Estimated cost in USD.
    :raises: ValueError if instance_type is not recognized.
    """
    if instance_type not in INSTANCE_COSTS:
        raise ValueError(
            f"Unknown instance type '{instance_type}'. Available types: {', '.join(INSTANCE_COSTS.keys())}.")
    info = INSTANCE_COSTS[instance_type]
    ops_per_sec = info["ops_per_sec"]
    cost_per_hour = info["cost_per_hour"]
    if ops_per_sec <= 0:
        # Avoid division by zero or nonsense
        logger.warning(f"Instance {instance_type} has invalid ops_per_sec value. Returning 0 cost.")
        return 0.0
    # Calculate time in seconds required
    seconds = operations / ops_per_sec
    # Calculate cost for that time
    cost = (seconds / 3600) * cost_per_hour
    logger.debug(f"Estimated {seconds:.2f} seconds on {instance_type} for {operations} ops, cost = ${cost:.6f}")
    return cost
