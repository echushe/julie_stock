import random
import torch
import torch.nn as nn
import copy

from train_daily_data.preprocess_of_batch import BatchPreprocessor
from train_daily_data.model_selection import resolve_gpu_plan


class ModelEnsemble(nn.Module):
    def __init__(self, models):
        
        super(ModelEnsemble, self).__init__()

        if isinstance(models, dict):
            # If models is a dictionary, convert it to a list
            models = list(models.values())
        
        self.models = torch.nn.ModuleList(models)
        self.n_ensemble = len(models)
        
    def forward(self, x):
        
        cls_out_list = []
        rate_out_list = []
        for model in self.models:
            cls, rate = model(x)
            cls_out_list.append(cls)
            rate_out_list.append(rate)

        # Concatenate the outputs from all models
        cls_out = torch.stack(cls_out_list, dim=0)  # Shape: (n_ensemble, batch_size, output_size)
        rate_out = torch.stack(rate_out_list, dim=0)  # Shape: (n_ensemble, batch_size, 1)

        return cls_out, rate_out  # cls_out: (n_ensemble, batch_size, output_size), rate_out: (n_ensemble, batch_size, 1)


class ModelEnsembleWithPrice(nn.Module):
    def __init__(self, models):
        
        super(ModelEnsembleWithPrice, self).__init__()

        if isinstance(models, dict):
            # If models is a dictionary, convert it to a list
            models = list(models.values())
        
        self.models = torch.nn.ModuleList(models)
        self.n_ensemble = len(models)
        
    def forward(self, x, last_price):
        
        cls_out_list = []
        rate_out_list = []
        for model in self.models:
            cls, rate = model(x, last_price)
            cls_out_list.append(cls)
            rate_out_list.append(rate)

        # Concatenate the outputs from all models
        cls_out = torch.stack(cls_out_list, dim=0)  # Shape: (n_ensemble, batch_size, output_size)
        rate_out = torch.stack(rate_out_list, dim=0)  # Shape: (n_ensemble, batch_size, 1)

        return cls_out, rate_out  # cls_out: (n_ensemble, batch_size, output_size), rate_out: (n_ensemble, batch_size, 1)


class ModelCluster:
    def __init__(self, models, config):
        self.voting = config['inference']['voting']

        if models is None:
            return

        if len(models) == 0:
            raise ValueError('Number of models must be greater than 0.')

        self.models_as_dict = models
        self.differenced = config['preprocessing']['differenced']

        gpu_plan = resolve_gpu_plan(n_models=len(models), config=config)
        for gpu_id, weight in gpu_plan:
            if weight > 0:
                # If the weight is greater than 0, it means this GPU can be used
                self.gpu_id = gpu_id
                break
        if len(models) == 1:
            # If there is only one model, use the same gpu as the model's
            if isinstance(models, dict):
                model = list(models.values())[0]
            elif isinstance(models, list):
                model = models[0]
            self.gpu_id = next(model.parameters()).device.index

        self.processor = BatchPreprocessor(
            normalize_method=config['preprocessing']['normalize_method'],
            differenced=self.differenced,
            ignore_turnover_data= config['preprocessing']['ignore_turnover_data'],)
        
    
    def determine_prediction_via_mean(self, cls_list):
        if len(cls_list) > 1:
            for i in range(len(cls_list)):
                cls_list[i] = cls_list[i].to('cuda:{}'.format(self.gpu_id))
        # cls_list is a list of tensors, each with shape (batch_size, n_classes)
        # Stack the tensors so that cls has shape (n_models, batch_size, n_classes)
        cls = torch.stack(cls_list, dim=0)
        # This is to smooth the cls output
        # Shape will change back to (batch_size, n_classes)
        cls = torch.mean(cls, dim=0)
        pred = cls.argmax(dim=1)

        return pred


    def determine_prediction_via_mean_fuse_neg_and_pos(self, cls_list):
        if len(cls_list) > 1:
            for i in range(len(cls_list)):
                cls_list[i] = cls_list[i].to('cuda:{}'.format(self.gpu_id))
        # cls_list is a list of tensors, each with shape (batch_size, n_classes)
        # Stack the tensors so that cls has shape (n_models, batch_size, n_classes)
        cls = torch.stack(cls_list, dim=0)
        n_classes = cls.shape[2]
        
        # This is to smooth the cls output
        # Shape will change back to (batch_size, n_classes)
        cls = torch.mean(cls, dim=0)

        # (batch_size, n_classes) to (batch_size)
        pred = cls.argmax(dim=1)

        mid_class = n_classes // 2
        # Fuse negative and positive classes
        cls[cls < mid_class] = 0
        cls[cls == mid_class] = 1
        cls[cls > mid_class] = 2

        return pred


    def determine_prediction_via_softmax_and_mean(self, cls_list):
        if len(cls_list) > 1:
            for i in range(len(cls_list)):
                cls_list[i] = cls_list[i].to('cuda:{}'.format(self.gpu_id))
        # cls_list is a list of tensors, each with shape (batch_size, n_classes)
        # Stack the tensors so that cls has shape (n_models, batch_size, n_classes)
        cls = torch.stack(cls_list, dim=0)
        
        # Apply softmax to cls along the last dimension (n_classes)
        # Shape will change to (n_models, batch_size, n_classes)
        cls = torch.softmax(cls, dim=2)

        # This is to smooth the cls output
        # Shape will change back to (batch_size, n_classes)
        cls = torch.mean(cls, dim=0)

        # Shape will change to (batch_size)
        pred = cls.argmax(dim=1)

        return pred


    def determine_prediction_via_softmax_and_mean_fuse_neg_and_pos(self, cls_list):
        if len(cls_list) > 1:
            for i in range(len(cls_list)):
                cls_list[i] = cls_list[i].to('cuda:{}'.format(self.gpu_id))
        # cls_list is a list of tensors, each with shape (batch_size, n_classes)
        # Stack the tensors so that cls has shape (n_models, batch_size, n_classes)
        cls = torch.stack(cls_list, dim=0)
        n_classes = cls.shape[2]
        
        # Apply softmax to cls along the last dimension (n_classes)
        # Shape will change to (n_models, batch_size, n_classes)
        cls = torch.softmax(cls, dim=2)

        # This is to smooth the cls output
        # Shape will change back to (batch_size, n_classes)
        cls = torch.mean(cls, dim=0)

        # Determine classification of each model
        # Shape will change to (batch_size)
        pred = cls.argmax(dim=1)

        mid_class = n_classes // 2
        # Fuse negative and positive classes
        pred[pred < mid_class] = 0
        pred[pred == mid_class] = 1
        pred[pred > mid_class] = 2

        return pred

    
    def determine_prediction_via_voting(self, cls_list):
        if len(cls_list) > 1:
            for i in range(len(cls_list)):
                cls_list[i] = cls_list[i].to('cuda:{}'.format(self.gpu_id))
        # cls_list is a list of tensors, each with shape (batch_size, n_classes)
        # Stack the tensors so that cls has shape (n_models, batch_size, n_classes)
        cls = torch.stack(cls_list, dim=0)

        if cls.dim() == 3:
            # Determine classification of each model
            # Shape will change to (n_models, batch_size)
            pred = cls.argmax(dim=2)
        elif cls.dim() == 2:
            # If cls is already in the shape (n_models, batch_size)
            pred = cls
        else:
            raise ValueError(f"cls_list should contain tensors with shape (batch_size, n_classes) or (batch_size,), but got {cls.shape}")
        
        # Vote alone the first dimension (n_models)
        # Shape will change to (batch_size)
        pred_of_most_vote = torch.mode(pred, dim=0)[0]
        
        # Count the number of votes for the most voted class
        # Shape will change to (batch_size)
        num_votes = torch.sum(pred == pred_of_most_vote, dim=0)
        rate_of_most_vote = num_votes / len(cls_list)

        return pred_of_most_vote, rate_of_most_vote


    def determine_prediction_via_voting_fuse_neg_and_pos(self, cls_list, n_classes=7):
        if len(cls_list) > 1:
            for i in range(len(cls_list)):
                cls_list[i] = cls_list[i].to('cuda:{}'.format(self.gpu_id))
        # cls_list is a list of tensors, each with shape (batch_size, n_classes)
        # Stack the tensors so that cls has shape (n_models, batch_size, n_classes)
        cls = torch.stack(cls_list, dim=0)
        
        if cls.dim() == 3:
            n_classes = cls.shape[2]
            # Determine classification of each model
            # Shape will change to (n_models, batch_size)
            pred = cls.argmax(dim=2)
        elif cls.dim() == 2:
            # If cls is already in the shape (n_models, batch_size)
            pred = cls
        else:
            raise ValueError(f"cls_list should contain tensors with shape (batch_size, n_classes) or (batch_size,), but got {cls.shape}")

        mid_class = n_classes // 2
        # Fuse negative and positive classes
        pred[pred < mid_class] = 0
        pred[pred == mid_class] = 1
        pred[pred > mid_class] = 2

        # Vote alone the first dimension (n_models)
        # Shape will change to (batch_size)
        pred_of_most_vote = torch.mode(pred, dim=0)[0]

        # Count the number of votes for the most voted class
        # Shape will change to (batch_size)
        num_votes = torch.sum(pred == pred_of_most_vote, dim=0)
        rate_of_most_vote = num_votes / len(cls_list)

        return pred_of_most_vote, rate_of_most_vote


    def dummy_predict(self, original_samples):
        # Dummy prediction for testing purposes
        pred_of_most_vote = []
        rate_of_most_vote = []
        pred_of_most_vote_3cls = []
        rate_of_most_vote_3cls = []

        pred = []
        pred_3cls = []

        for _ in original_samples:
            pred_of_most_vote.append(random.randint(0, 6))
            rate_of_most_vote.append(random.uniform(0.5, 1.0))
            pred_of_most_vote_3cls.append(random.randint(0, 2))
            rate_of_most_vote_3cls.append(random.uniform(0.5, 1.0))

            pred.append(random.randint(0, 6))
            pred_3cls.append(random.randint(0, 2))

        if self.voting:
            # If voting is enabled, return the dummy predictions
            return pred_of_most_vote, rate_of_most_vote, pred_of_most_vote_3cls, rate_of_most_vote_3cls
        else:
            # If voting is not enabled, return the dummy predictions
            return pred, pred_3cls
        
    
    def predict(self, original_samples, infer_batch_size=1024):
        return self.dummy_predict(original_samples)


class CLSModelCluster (ModelCluster):
    def __init__(self, models, config):
        super().__init__(models, config)


    def model_forward(self, model, infer_batch_size, inputs, last_price=None):

        cls_list = []
        rate_list = []

        if len(self.models_as_dict) > 1:
            # incase gpu of inputs and last_price is different from the model's gpu
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            if last_price is not None:
                last_price = last_price.to(device)

        for i in range(0, inputs.shape[0], infer_batch_size):
            inputs_batch = inputs[i : i + infer_batch_size]
            if last_price is not None:
                last_price_batch = last_price[i : i + infer_batch_size]
                cls, rate = model(inputs_batch, last_price_batch)
            else:
                cls, rate = model(inputs_batch)

            cls_list.append(cls)
            rate_list.append(rate)

        cls = torch.cat(cls_list, dim=0)
        rate = torch.cat(rate_list, dim=0)
        return cls, rate


    def predict(self, original_samples, infer_batch_size=1024):

        indices_input_batch = []
        stock_input_batch = []

        for indices_input_array, stock_input_array in original_samples:

            indices_input_batch.append(torch.from_numpy(indices_input_array))
            stock_input_batch.append(torch.from_numpy(stock_input_array))

        indices_input_batch = torch.stack(indices_input_batch, dim=0).to('cuda:{}'.format(self.gpu_id))
        stock_input_batch = torch.stack(stock_input_batch, dim=0).to('cuda:{}'.format(self.gpu_id))

        if self.differenced:

            indices_input_batch_n, stock_input_batch_n, indices_last_price_batch_n, stock_last_price_batch_n = \
                self.processor.process_sample(indices_input_batch, stock_input_batch)
            
            #print("indices_input_batch shape: ", indices_input_batch.shape)
            #print("stock_input_batch shape: ", stock_input_batch.shape)
            #print("indices_last_price_batch shape: ", indices_last_price_batch.shape)
            #print("stock_last_price_batch shape: ", stock_last_price_batch.shape)

            last_price = torch.cat((indices_last_price_batch_n, stock_last_price_batch_n), dim=1)
        else:
            indices_input_batch_n, stock_input_batch_n = \
                self.processor.process_sample(indices_input_batch, stock_input_batch)

        inputs = torch.cat((indices_input_batch_n, stock_input_batch_n), dim=2)

        cls_list = []

        for time_as_key, model in sorted(self.models_as_dict.items(), key=lambda x: x[0]):
            if self.differenced:
                #cls, rate = model(inputs, last_price)
                cls, rate = self.model_forward(model, infer_batch_size, inputs, last_price)
            else:
                #cls, rate = model(inputs)
                cls, rate = self.model_forward(model, infer_batch_size, inputs)
            cls_list.append(cls)

        if self.voting:
            # Get final prediction via voting along the first dimension (n_models)
            pred_of_most_vote, rate_of_most_vote = self.determine_prediction_via_voting(cls_list)

            # Fuse classes into 3 classes: negative, neutral, positive
            # Get final prediction via voting along the first dimension (n_models)
            pred_of_most_vote_3cls, rate_of_most_vote_3cls = self.determine_prediction_via_voting_fuse_neg_and_pos(cls_list)

            # (batch_size,), (batch_size,), (batch_size,), (batch_size,)
            return pred_of_most_vote, rate_of_most_vote, pred_of_most_vote_3cls, rate_of_most_vote_3cls
        else:
            # Get final prediction via mean along the first dimension (n_models)
            pred = self.determine_prediction_via_softmax_and_mean(cls_list)

            # Fuse classes into 3 classes: negative, neutral, positive
            pred_3cls = self.determine_prediction_via_softmax_and_mean_fuse_neg_and_pos(cls_list)

            # (batch_size,), (batch_size,)
            return pred, pred_3cls


class REGModelCluster (ModelCluster):
    def __init__(self, models, config):   
        super().__init__(models, config)

    def model_forward(self, model, infer_batch_size, inputs, last_price=None):

        pred_list = []

        if len(self.models_as_dict) > 1:
            # incase gpu of inputs and last_price is different from the model's gpu
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            if last_price is not None:
                last_price = last_price.to(device)

        for i in range(0, inputs.shape[0], infer_batch_size):
            inputs_batch = inputs[i : i + infer_batch_size]
            if last_price is not None:
                last_price_batch = last_price[i : i + infer_batch_size]
                pred = model(inputs_batch, last_price_batch)
            else:
                pred = model(inputs_batch)

            pred_list.append(pred)

        pred = torch.cat(pred_list, dim=0)

        return pred


    def rate_to_cls(self, rate):
        boundaries = torch.tensor([-1e-5, 1e-5], dtype=torch.float32).to('cuda:{}'.format(self.gpu_id))
        return torch.bucketize(rate, boundaries, right=True).to(torch.int64)


    def predict(self, original_samples, infer_batch_size=1024):

        indices_input_batch =[]
        stock_input_batch = []

        for indices_input_array, stock_input_array in original_samples:

            indices_input_batch.append(torch.from_numpy(indices_input_array))
            stock_input_batch.append(torch.from_numpy(stock_input_array))

        indices_input_batch = torch.stack(indices_input_batch, dim=0).to('cuda:{}'.format(self.gpu_id))
        stock_input_batch = torch.stack(stock_input_batch, dim=0).to('cuda:{}'.format(self.gpu_id))

        if self.differenced:

            indices_input_batch_n, stock_input_batch_n, indices_last_price_batch_n, stock_last_price_batch_n = \
                self.processor.process_sample(indices_input_batch, stock_input_batch)
            
            #print("indices_input_batch shape: ", indices_input_batch.shape)
            #print("stock_input_batch shape: ", stock_input_batch.shape)
            #print("indices_last_price_batch shape: ", indices_last_price_batch.shape)
            #print("stock_last_price_batch shape: ", stock_last_price_batch.shape)

            last_price = torch.cat((indices_last_price_batch_n, stock_last_price_batch_n), dim=1)
        else:
            indices_input_batch_n, stock_input_batch_n = \
                self.processor.process_sample(indices_input_batch, stock_input_batch)

        inputs = torch.cat((indices_input_batch_n, stock_input_batch_n), dim=2)

        rate_list = []
        
        for time_as_key, model in sorted(self.models_as_dict.items(), key=lambda x: x[0]):
            if self.differenced:
                #pred = model(inputs, last_price)
                pred = self.model_forward(model, infer_batch_size, inputs, last_price)
            else:
                #pred = model(inputs)
                pred = self.model_forward(model, infer_batch_size, inputs)

            rate = self.processor.calculate_price_change_rate_via_differenced_data(last_price, pred)
            rate_list.append(rate)

        if self.voting:
            cls_list = []
            for rate in rate_list:
                # Convert the rate to a classification
                # classify rate into 3 classes: 0, 1, 2
                # rate is a tensor of shape (batch_size,)
                # 0: negative < -0.01, 1: neutral [-0.01, 0.01], 2: positive > 0.01
                pred_cls = self.rate_to_cls(rate)
                cls_list.append(pred_cls)

            # Get final prediction via voting along the first dimension (n_models)
            pred_of_most_vote, rate_of_most_vote = self.determine_prediction_via_voting(cls_list)

            # Fuse classes into 3 classes: negative, neutral, positive
            # Get final prediction via voting along the first dimension (n_models)
            pred_of_most_vote_3cls, rate_of_most_vote_3cls = self.determine_prediction_via_voting_fuse_neg_and_pos(cls_list, n_classes=3)

            # (batch_size,), (batch_size,), (batch_size,), (batch_size,)
            return pred_of_most_vote, rate_of_most_vote, pred_of_most_vote_3cls, rate_of_most_vote_3cls
        else:
            for rate in rate_list:
                rate = rate.to('cuda:{}'.format(self.gpu_id))
            rate_array = torch.stack(rate_list, dim=0)
            rate_mean = rate_array.mean(dim=0)
            pred_cls = self.rate_to_cls(rate_mean)
            pred_3cls = pred_cls.clone()

            # (batch_size,), (batch_size,),
            return pred_cls, pred_3cls
