import numpy as np

class EarningRateCLSRule:
    """
    Class to define the rules for classifying price movements.
    """

    def __init__(self, cls_threshold_list: list = [0.0,]):
        """
        Initialize the PriceCLSRule with a list of thresholds.
        For example: [-1.5, -0.5, 0.4, 1.4, 2.5, 4.3]
        """
        for i in range(1, len(cls_threshold_list)):
            if cls_threshold_list[i] <= cls_threshold_list[i - 1]:
                raise ValueError("Thresholds must be in ascending order.")
        
        self.cls_threshold_list = cls_threshold_list

    def classify(self, input_array, target_array) -> tuple:
        """
        Classify the price change based on the defined thresholds.

        """
        price_change = (target_array[-1, 1] - input_array[-1, 1]) / (input_array[-1, 1] + 1e-20)

        # Check if the price change is within the defined thresholds
        category = 0
        while category < len(self.cls_threshold_list):
            
            if price_change < self.cls_threshold_list[category]:
                return category, price_change     
            category += 1

        return category, price_change
    
    def price_change_rate(self, input_array, target_array) -> float:
        """
        Calculate the price change rate based on the defined thresholds.

        """
        price_change = (target_array[-1, 1] - input_array[-1, 1]) / (input_array[-1, 1] + 1e-20)
        price_change = max(min(price_change, 100.0), -100.0) # Clamp the value between -100 and 100

        return price_change