import torch
import torch.utils.data as data
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict
import os
import random
from typing import Dict, List, Tuple, Union, Optional


class IndexedDataset(data.Dataset):
    """Dataset wrapper that returns indices along with data."""
    
    def __init__(self, dataset: data.Dataset):
        self.dataset = dataset
    
    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int]:
        data_item, target = self.dataset[index]
        return index, data_item, target
    
    def __len__(self) -> int:
        return len(self.dataset)


class IndexedFilteredDataset(data.Dataset):
    """Dataset wrapper that filters samples and returns original indices"""
    def __init__(self, dataset, selected_indices):
        """
        Args:
            dataset: Original dataset (must be indexed, i.e., returns (index, data, target))
            selected_indices: List of indices to keep (filtered samples)
        """
        self.dataset = dataset
        self.selected_indices = sorted(selected_indices)  # Sort for consistent ordering
        
        # Verify that dataset is indexed (returns tuples with len=3)
        self._verify_indexed_dataset()
        
        # Create mapping from new indices to original indices
        self.index_mapping = {new_idx: orig_idx for new_idx, orig_idx in enumerate(self.selected_indices)}
    
    def _verify_indexed_dataset(self):
        """Verify that the dataset returns indexed data with len=3"""
        try:
            sample = self.dataset[0]
            if not isinstance(sample, tuple) or len(sample) != 3:
                raise ValueError(f"Dataset must return indexed data with len=3, got {type(sample)} with len={len(sample) if isinstance(sample, tuple) else 'N/A'}")
            if not isinstance(sample[0], int):
                raise ValueError(f"First element of dataset sample must be an integer index, got {type(sample[0])}")
        except Exception as e:
            raise ValueError(f"Failed to verify indexed dataset: {e}")
    
    def __getitem__(self, index):
        """Get item from the filtered dataset with original index"""
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        
        # Get the original index
        original_index = self.index_mapping[index]
        
        # Get the original data item (guaranteed to be (index, data, target) format)
        data_item = self.dataset[original_index]
        
        # Extract data from indexed format (index, data, target)
        orig_idx, img, target = data_item[0], data_item[1], data_item[2]
        
        return original_index, img, target
    
    def __len__(self):
        """Return the size of the filtered dataset"""
        return len(self.selected_indices)
    
    def get_original_index(self, new_index):
        """Get the original dataset index for a given new index"""
        return self.index_mapping[new_index]
    
    def get_filter_info(self):
        """Get information about the filtered dataset"""
        return {
            'original_size': len(self.dataset),
            'filtered_size': len(self.selected_indices),
            'selected_indices': self.selected_indices
        }


class IndexedRepeatIndicesDataset(data.Dataset):
    """Dataset wrapper that repeats specific samples and returns original indices"""
    def __init__(self, dataset, repeat_indices, repeat_factor, add_noise=False, noise_std=0.01):
        """
        Args:
            dataset: Original dataset (must be indexed, i.e., returns (index, data, target))
            repeat_indices: List of indices to repeat
            repeat_factor: Number of times to repeat each specified sample
            add_noise: Whether to add Gaussian noise to repeated samples (additive)
            noise_std: Standard deviation of Gaussian noise
        """
        self.dataset = dataset
        self.repeat_indices = set(repeat_indices)  # Convert to set for O(1) lookup
        self.repeat_factor = repeat_factor
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        # Verify that dataset is indexed (returns tuples with len=3)
        self._verify_indexed_dataset()
        
        # Calculate new dataset size
        self.original_size = len(dataset)
        self.repeated_samples = len(repeat_indices) * repeat_factor
        self.new_size = self.original_size + self.repeated_samples
        
        # Create mapping from new indices to original indices
        self.index_mapping = self._create_index_mapping()
    
    def _verify_indexed_dataset(self):
        """Verify that the dataset returns indexed data with len=3"""
        try:
            sample = self.dataset[0]
            if not isinstance(sample, tuple) or len(sample) != 3:
                raise ValueError(f"Dataset must return indexed data with len=3, got {type(sample)} with len={len(sample) if isinstance(sample, tuple) else 'N/A'}")
            if not isinstance(sample[0], int):
                raise ValueError(f"First element of dataset sample must be an integer index, got {type(sample[0])}")
        except Exception as e:
            raise ValueError(f"Failed to verify indexed dataset: {e}")
    
    def _create_index_mapping(self):
        """Create mapping from new dataset indices to original dataset indices"""
        mapping = {}
        new_idx = 0
        
        # First, map all original samples
        for orig_idx in range(self.original_size):
            mapping[new_idx] = orig_idx
            new_idx += 1
        
        # Then, add repeated samples
        for orig_idx in self.repeat_indices:
            for _ in range(self.repeat_factor):
                mapping[new_idx] = orig_idx
                new_idx += 1
        
        return mapping
    
    def __getitem__(self, index):
        """Get item from the extended dataset with original index"""
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        
        # Get the original index
        original_index = self.index_mapping[index]
        
        # Get the original data item (guaranteed to be (index, data, target) format)
        data_item = self.dataset[original_index]
        
        # Check if this is a repeated sample and we need to add noise
        is_repeated = index >= self.original_size
        
        # Extract data from indexed format (index, data, target)
        orig_idx, img, target = data_item[0], data_item[1], data_item[2]
        
        # Add noise to repeated samples if requested
        if is_repeated and self.add_noise:
            # Additive Gaussian noise: x + N(0, σ²)
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise
            # Clamp to valid range [0, 1]
            img = torch.clamp(img, 0, 1)
        
        return original_index, img, target
    
    def __len__(self):
        """Return the new size of the dataset"""
        return self.new_size
    
    def get_original_index(self, new_index):
        """Get the original dataset index for a given new index"""
        return self.index_mapping[new_index]
    
    def get_repeat_info(self):
        """Get information about the repeated samples"""
        return {
            'original_size': self.original_size,
            'new_size': self.new_size,
            'repeated_samples': self.repeated_samples,
            'repeat_indices': list(self.repeat_indices),
            'repeat_factor': self.repeat_factor
        }


def match(D: Dict, indices: Union[List, torch.Tensor], save_item: torch.Tensor):
    """Map batch results to individual samples using path/indices"""
    for i in range(len(indices)):
        # Convert tensor index to integer if needed
        key = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
        D[key] = save_item[i]
    return


def get_loss_and_grad_for_batch(net, criterion, batch_data, batch_labels, temp=1):
    was_training = net.training
    net.eval()
    # Ensure batch_data is a fresh tensor that requires gradients
    batch_data = batch_data.detach().clone().requires_grad_(True)
    outputs = net(batch_data)
    loss = criterion(outputs / temp, batch_labels)
    grad = torch.autograd.grad(loss.sum(), batch_data, retain_graph=True)[0]
    loss_grad = grad.reshape(grad.size(0), -1).norm(dim=1).cpu().detach()

    net.zero_grad()
    if batch_data.grad is not None:
        batch_data.grad.zero_()

    if was_training:
        net.train()

    return loss, loss_grad, torch.nn.functional.softmax(outputs, 1).max(1)[0]


def get_loss_for_batch(net, criterion, batch_data, batch_labels, temp=1):
    net.eval()
    outputs = net(batch_data)
    loss = criterion(outputs / temp, batch_labels)
    return loss


def get_regularized_curvature_for_batch(net, criterion, batch_data, batch_labels, h=1e-3, niter=10, temp=1):
    num_samples = batch_data.shape[0]
    net.eval()
    regr = torch.zeros(num_samples)
    eigs = torch.zeros(num_samples)
    for _ in range(niter):
        v = torch.randint_like(batch_data, high=2).cuda()
        # Generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        v = h * (v + 1e-7)

        # Ensure batch_data is a fresh tensor that requires gradients
        batch_data = batch_data.detach().clone().requires_grad_(True)
        outputs_pos = net(batch_data + v)
        outputs_orig = net(batch_data)
        loss_pos = criterion(outputs_pos / temp, batch_labels).sum()
        loss_orig = criterion(outputs_orig / temp, batch_labels).sum()
        grad_diff = torch.autograd.grad((loss_pos - loss_orig), batch_data)[0]

        regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()
        eigs += torch.diag(torch.matmul(v.reshape(num_samples, -1), grad_diff.reshape(num_samples, -1).T)).cpu().detach()
        net.zero_grad()
        if batch_data.grad is not None:
            batch_data.grad.zero_()

    curv_estimate = regr / niter
    return curv_estimate


def compute_curvature_for_dataset(model, criterion, train_loader, dataset_name, model_name, 
                                 device='cuda', save_dir='/home/nano01/a/sharm444/Robust_Dataset_Project/scores/computed_curvature/', 
                                 use_paths=False, batch_size=None, debug=False, split='train'):
    """Compute curvature for entire dataset and save results
    
    Args:
        split: 'train' or 'val' to specify which split is being computed (default: 'train')
    """

    # Create indexed dataloader
    indexed_loader, indexed_dataset = create_indexed_and_repeated_dataloader(
        train_loader, [], 0, use_paths, debug=debug
    )

    # Initialize curvature dictionary
    curvature = {}

    print(f"Computing curvature for {len(indexed_dataset)} samples...")

    # Calculate curvature for each batch
    for batch_idx, (indices, inputs, targets) in enumerate(indexed_loader):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{len(indexed_loader)}")
        
        inputs, targets = inputs.to(device), targets.to(device)
        curv_estimate = get_regularized_curvature_for_batch(
            model, criterion, inputs, targets
        )
        curv_estimate = torch.log(curv_estimate)
        
        # Map results to indices
        match(curvature, indices, curv_estimate)

    if debug:
        # Check the size of the curvature dictionary
        print(f"Curvature dictionary size: {len(curvature)}")
        print(f"Indexed dataset size: {len(indexed_dataset)}")
        
        # Debug: Print first few curvature values safely
        print("First few curvature values:")
        for i in range(min(10, len(curvature))):
            if i in curvature:
                print(f"Index {i}: {curvature[i]}")
            else:
                print(f"Index {i}: Not found in curvature dictionary")
        
        # Also print some actual keys that exist
        actual_keys = list(curvature.keys())[:5]
        print(f"Sample actual keys in curvature: {actual_keys}")
        
        if len(curvature) != len(indexed_dataset):
            print(f"WARNING: Expected {len(indexed_dataset)} samples, got {len(curvature)}")

    # Convert curvature dictionary to regular Python dict with int keys and float values
    regular_curvature = {}
    for key, value in curvature.items():
        # Convert key to int if it's a tensor
        int_key = key.item() if torch.is_tensor(key) else int(key)
        # Convert value to float if it's a tensor
        float_value = value.item() if torch.is_tensor(value) else float(value)
        regular_curvature[int_key] = float_value

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    curvature_file_name = os.path.join(save_dir, f'{model_name}_{dataset_name}_{split}_curvature.npy')
    np.save(curvature_file_name, regular_curvature)
    
    print(f"Curvature computation completed. Saved to {curvature_file_name}")
    return curvature_file_name, regular_curvature


def create_indexed_dataloader(original_loader, debug=False):
    """Create a dataloader with indexing for all samples"""
    original_dataset = original_loader.dataset
    final_dataset = IndexedDataset(original_dataset)
    
    return _create_dataloader_from_dataset(original_loader, final_dataset, debug, "indexed")


def create_filtered_dataloader(original_loader, selected_indices, debug=False):
    """Create a dataloader with filtering based on selected indices"""
    original_dataset = original_loader.dataset
    indexed_dataset = IndexedDataset(original_dataset)
    final_dataset = IndexedFilteredDataset(indexed_dataset, selected_indices)
    
    return _create_dataloader_from_dataset(original_loader, final_dataset, debug, "filtered")


def create_repeated_dataloader(original_loader, repeat_indices, repeat_factor, add_noise=False, noise_std=0.01, debug=False):
    """Create a dataloader with repetition of specified indices"""
    original_dataset = original_loader.dataset
    indexed_dataset = IndexedDataset(original_dataset)
    final_dataset = IndexedRepeatIndicesDataset(
        indexed_dataset, 
        repeat_indices, 
        repeat_factor,
        add_noise=add_noise,
        noise_std=noise_std
    )
    
    return _create_dataloader_from_dataset(original_loader, final_dataset, debug, "repeated")


def create_filtered_and_repeated_dataloader(original_loader, filter_indices, repeat_indices, repeat_factor, add_noise=False, noise_std=0.01, debug=False):
    """Create a dataloader that first filters, then repeats specified indices"""
    original_dataset = original_loader.dataset
    
    # First create filtered dataset
    filtered_dataset = IndexedFilteredDataset(original_dataset, filter_indices)
    
    # Then create repeated dataset from the filtered dataset
    final_dataset = IndexedRepeatIndicesDataset(
        filtered_dataset, 
        repeat_indices, 
        repeat_factor,
        add_noise=add_noise,
        noise_std=noise_std
    )
    
    return _create_dataloader_from_dataset(original_loader, final_dataset, debug, "filtered_and_repeated")


def create_indexed_and_repeated_dataloader(original_loader, repeat_indices, repeat_factor, use_paths=False, force_sequential=False, debug=False, add_noise=False, noise_std=0.01):
    """Legacy function - now redirects to appropriate new function"""
    if repeat_factor == 0:
        if len(repeat_indices) == 0:
            return create_indexed_dataloader(original_loader, debug)
        else:
            return create_filtered_dataloader(original_loader, repeat_indices, debug)
    else:
        return create_repeated_dataloader(original_loader, repeat_indices, repeat_factor, add_noise, noise_std, debug)


def _create_dataloader_from_dataset(original_loader, final_dataset, debug=False, operation_type="unknown"):
    """Helper function to create dataloader from a dataset with proper configuration"""
    original_dataset = original_loader.dataset
    
    # Get the original dataloader parameters
    batch_size = original_loader.batch_size
    num_workers = original_loader.num_workers
    pin_memory = original_loader.pin_memory
    original_sampler = original_loader.sampler
    drop_last = original_loader.drop_last
    
    if debug:
        # Debug original dataloader info
        print(f"\nOriginal dataloader debug info:")
        print(f"  Original dataset size: {len(original_dataset)}")
        print(f"  Original dataloader length: {len(original_loader)}")
        print(f"  Original batch_size: {batch_size}")
        print(f"  Inferred drop_last: {drop_last}")
        print(f"  Original sampler: {type(original_sampler).__name__ if original_sampler else 'None'}")
    
    # Check if the final dataset is empty
    if len(final_dataset) == 0:
        raise ValueError(f"Cannot create dataloader for empty dataset. Dataset size: {len(final_dataset)}")
    
    # Handle distributed sampler properly
    if original_sampler is not None and hasattr(original_sampler, 'num_replicas'):
        # This is a DistributedSampler, create a new one for the expanded dataset
        import torch.utils.data.distributed as distributed
        new_sampler = distributed.DistributedSampler(
            final_dataset,
            num_replicas=original_sampler.num_replicas,
            rank=original_sampler.rank,
            shuffle=original_sampler.shuffle,
            drop_last=drop_last
        )
        if debug:
            print(f"Created new DistributedSampler for {operation_type} dataset:")
            print(f"  Original sampler num_replicas: {original_sampler.num_replicas}")
            print(f"  Original sampler rank: {original_sampler.rank}")
            print(f"  New sampler num_replicas: {new_sampler.num_replicas}")
            print(f"  New sampler rank: {new_sampler.rank}")
            print(f"  New sampler length: {len(new_sampler)}")
    else:
        # Create a new sampler for the expanded dataset instead of reusing the original
        # The original sampler was designed for the smaller dataset and won't access the new samples
        if original_sampler is not None:
            # If there was an original sampler, create a new one of the same type
            if hasattr(original_sampler, 'shuffle'):
                shuffle = original_sampler.shuffle
            else:
                # Default to True for RandomSampler, False for SequentialSampler
                shuffle = isinstance(original_sampler, torch.utils.data.RandomSampler)
            
            new_sampler = torch.utils.data.RandomSampler(final_dataset) if shuffle else torch.utils.data.SequentialSampler(final_dataset)
            if debug:
                print(f"Created new sampler for {operation_type} dataset:")
                print(f"  Original sampler type: {type(original_sampler).__name__}")
                print(f"  New sampler type: {type(new_sampler).__name__}")
                print(f"  Shuffle: {shuffle}")
        else:
            # No original sampler, use None (will use shuffle=True)
            new_sampler = None
            if debug:
                print(f"No original sampler, will use shuffle=True")
    
    # Create new data loader
    final_loader = data.DataLoader(
        final_dataset, 
        batch_size=batch_size,
        shuffle=(new_sampler is None),  # Only shuffle if no sampler
        num_workers=num_workers,
        sampler=new_sampler,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    # Print information about the dataset
    if hasattr(final_dataset, 'get_filter_info'):
        # Filtering case
        filter_info = final_dataset.get_filter_info()
        if debug:
            print(f"Created {operation_type} dataset:")
            print(f"  Original size: {filter_info['original_size']}")
            print(f"  Final size: {filter_info['filtered_size']}")
            print(f"  Number of selected indices: {len(filter_info['selected_indices'])}")
            
            # Add debugging information about batches
            print(f"\nBatch information:")
            print(f"  Batch size: {batch_size}")
            print(f"  Drop last: {drop_last}")
            print(f"  Original batches: {filter_info['original_size'] // batch_size} (complete)")
            print(f"  Final batches: {filter_info['filtered_size'] // batch_size} (complete)")
            print(f"  Actual dataloader length: {len(final_loader)}")
            print(f"  Expected dataloader length: {filter_info['filtered_size'] // batch_size}")
            
            if drop_last:
                remaining_samples = filter_info['filtered_size'] % batch_size
                if remaining_samples > 0:
                    print(f"  Note: {remaining_samples} samples dropped due to drop_last=True")
    elif hasattr(final_dataset, 'get_repeat_info'):
        # Repetition case
        repeat_info = final_dataset.get_repeat_info()
        if debug:
            print(f"Created {operation_type} dataset:")
            print(f"  Original size: {repeat_info['original_size']}")
            print(f"  New size: {repeat_info['new_size']}")
            print(f"  Repeated samples: {repeat_info['repeated_samples']}")
            print(f"  Repeat factor: {repeat_info['repeat_factor']}")
            print(f"  Number of unique indices repeated: {len(repeat_info['repeat_indices'])}")
            
            # Add debugging information about batches
            print(f"\nBatch information:")
            print(f"  Batch size: {batch_size}")
            print(f"  Drop last: {drop_last}")
            print(f"  Original batches: {repeat_info['original_size'] // batch_size} (complete)")
            print(f"  New batches: {repeat_info['new_size'] // batch_size} (complete)")
            print(f"  Actual dataloader length: {len(final_loader)}")
            print(f"  Expected dataloader length: {repeat_info['new_size'] // batch_size}")
            
            if drop_last:
                remaining_samples = repeat_info['new_size'] % batch_size
                if remaining_samples > 0:
                    print(f"  Note: {remaining_samples} samples dropped due to drop_last=True")
    
    return final_loader, final_dataset


class CurriculamSampler(Sampler):
    """
    Enhanced curriculum sampler with multiple biasing strategies.
    
    Biasing strategies:
    - 'default': Uniform random sampling
    - 'progressive_favor': Gradually favor important indices as training progresses
    - 'progressive_against': Gradually favor non-important indices as training progresses
    - 'switch_favor': Favor non-important indices before switch_epoch, then favor important indices
    - 'cyclic': Alternate between favoring important and non-important indices
    - 'exponential': Use exponential weighting based on importance
    - 'score_adaptive': Adapt weights based on score values (if provided)
    """
    
    def __init__(self, dataset_size, important_indices, total_epochs, current_epoch=0,
                 strategy='progressive_favor', switch_epoch=None, bias_strength=1.0,
                 score_dict=None, min_weight=0.1, max_weight=10.0):
        """
        Args:
            dataset_size: Total number of samples in dataset
            important_indices: Set of indices to bias towards/against
            total_epochs: Total number of training epochs
            current_epoch: Current epoch (0-indexed)
            strategy: Biasing strategy ('default', 'progressive_favor', 'progressive_against', 
                     'switch_favor', 'cyclic', 'exponential', 'score_adaptive')
            switch_epoch: Epoch at which to switch bias direction (for 'switch_favor' strategy)
            bias_strength: Strength of the bias (multiplier for weight differences)
            score_dict: Dictionary mapping indices to score values (for 'score_adaptive')
            min_weight: Minimum weight for any sample
            max_weight: Maximum weight for any sample
        """
        self.dataset_size = dataset_size
        self.all_indices = list(range(dataset_size))
        self.important_indices = set(important_indices)
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch
        self.strategy = strategy
        self.switch_epoch = switch_epoch if switch_epoch is not None else total_epochs // 2
        self.bias_strength = bias_strength
        self.score_dict = score_dict
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Validate strategy
        valid_strategies = ['default', 'progressive_favor', 'progressive_against', 
                          'switch_favor', 'cyclic', 'exponential', 'score_adaptive']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")

    def update_epoch(self, epoch):
        """Update the current epoch"""
        self.current_epoch = epoch

    def _compute_weights(self):
        """Compute weights based on the selected strategy"""
        progress = self.current_epoch / self.total_epochs
        
        if self.strategy == 'default':
            # Uniform weights
            return [1.0] * self.dataset_size
            
        elif self.strategy == 'progressive_favor':
            # Gradually favor important indices
            bias_weight = min(1.0, progress) * self.bias_strength
            weights = []
            for idx in self.all_indices:
                if idx in self.important_indices:
                    weights.append(1.0 + bias_weight)
                else:
                    weights.append(1.0)
            return weights
            
        elif self.strategy == 'progressive_against':
            # Gradually favor non-important indices
            bias_weight = min(1.0, progress) * self.bias_strength
            weights = []
            for idx in self.all_indices:
                if idx in self.important_indices:
                    weights.append(1.0)
                else:
                    weights.append(1.0 + bias_weight)
            return weights
            
        elif self.strategy == 'switch_favor':
            # Favor non-important before switch_epoch, then favor important
            if self.current_epoch < self.switch_epoch:
                # Before switch: favor non-important
                switch_progress = self.current_epoch / self.switch_epoch
                bias_weight = min(1.0, switch_progress) * self.bias_strength
                weights = []
                for idx in self.all_indices:
                    if idx in self.important_indices:
                        weights.append(1.0)
                    else:
                        weights.append(1.0 + bias_weight)
            else:
                # After switch: favor important
                remaining_epochs = self.total_epochs - self.switch_epoch
                current_remaining = self.current_epoch - self.switch_epoch
                switch_progress = current_remaining / remaining_epochs
                bias_weight = min(1.0, switch_progress) * self.bias_strength
                weights = []
                for idx in self.all_indices:
                    if idx in self.important_indices:
                        weights.append(1.0 + bias_weight)
                    else:
                        weights.append(1.0)
            return weights
            
        elif self.strategy == 'cyclic':
            # Alternate between favoring important and non-important
            cycle_length = self.total_epochs // 4  # Complete cycle every 1/4 of training
            cycle_progress = (self.current_epoch % cycle_length) / cycle_length
            if self.current_epoch // cycle_length % 2 == 0:
                # Favor important indices
                bias_weight = cycle_progress * self.bias_strength
                weights = []
                for idx in self.all_indices:
                    if idx in self.important_indices:
                        weights.append(1.0 + bias_weight)
                    else:
                        weights.append(1.0)
            else:
                # Favor non-important indices
                bias_weight = cycle_progress * self.bias_strength
                weights = []
                for idx in self.all_indices:
                    if idx in self.important_indices:
                        weights.append(1.0)
                    else:
                        weights.append(1.0 + bias_weight)
            return weights
            
        elif self.strategy == 'exponential':
            # Use exponential weighting based on importance
            base_weight = 1.0
            important_weight = base_weight * np.exp(self.bias_strength * progress)
            weights = []
            for idx in self.all_indices:
                if idx in self.important_indices:
                    weights.append(important_weight)
                else:
                    weights.append(base_weight)
            return weights
            
        elif self.strategy == 'score_adaptive':
            # Adapt weights based on score values
            if self.score_dict is None:
                raise ValueError("score_dict must be provided for 'score_adaptive' strategy")
            
            # Normalize score values
            scores = [self.score_dict.get(idx, 0.0) for idx in self.all_indices]
            min_score = min(scores) if scores else 0.0
            max_score = max(scores) if scores else 1.0
            if max_score > min_score:
                normalized_scores = [(c - min_score) / (max_score - min_score) for c in scores]
            else:
                normalized_scores = [0.5] * len(scores)
            
            # Apply adaptive weighting
            weights = []
            for i, idx in enumerate(self.all_indices):
                if idx in self.important_indices:
                    # Important indices get score-based weight
                    weight = 1.0 + self.bias_strength * normalized_scores[i] * progress
                else:
                    # Non-important indices get inverse score weight
                    weight = 1.0 + self.bias_strength * (1.0 - normalized_scores[i]) * progress
                weights.append(weight)
            return weights
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _normalize_weights(self, weights):
        """Normalize weights to be within min_weight and max_weight bounds"""
        min_w = min(weights)
        max_w = max(weights)
        
        if max_w > min_w:
            # Normalize to [0, 1] then scale to [min_weight, max_weight]
            normalized = [(w - min_w) / (max_w - min_w) for w in weights]
            scaled = [self.min_weight + (self.max_weight - self.min_weight) * n for n in normalized]
        else:
            # All weights are the same, set to middle value
            scaled = [(self.min_weight + self.max_weight) / 2] * len(weights)
        
        return scaled

    def __iter__(self):
        """Generate indices based on the selected strategy"""
        weights = self._compute_weights()
        weights = self._normalize_weights(weights)
        
        sampled_indices = random.choices(self.all_indices, weights=weights, k=self.dataset_size)
        return iter(sampled_indices)

    def __len__(self):
        return self.dataset_size
    
    def get_strategy_info(self):
        """Get information about the current sampling strategy"""
        weights = self._compute_weights()
        weights = self._normalize_weights(weights)
        
        important_weights = [weights[i] for i, idx in enumerate(self.all_indices) if idx in self.important_indices]
        other_weights = [weights[i] for i, idx in enumerate(self.all_indices) if idx not in self.important_indices]
        
        return {
            'strategy': self.strategy,
            'current_epoch': self.current_epoch,
            'progress': self.current_epoch / self.total_epochs,
            'avg_important_weight': np.mean(important_weights) if important_weights else 0.0,
            'avg_other_weight': np.mean(other_weights) if other_weights else 0.0,
            'weight_ratio': np.mean(important_weights) / np.mean(other_weights) if other_weights and np.mean(other_weights) > 0 else float('inf'),
            'min_weight': min(weights),
            'max_weight': max(weights)
        }