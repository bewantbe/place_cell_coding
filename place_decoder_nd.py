import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod

class PlaceFieldSystemBase(ABC):
    def __init__(self, n_cells_per_dim, pos_range=(-1, 1)):
        """Initialize base place field system.
        
        Args:
            n_cells_per_dim: Number of cells along each dimension
            pos_range: (min, max) range of positions in each dimension
        """
        self.n_cells_per_dim = n_cells_per_dim
        self.pos_range = pos_range
        # Set GPU device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set width of Gaussian response (adjust this to control overlap)
        self.sigma = (pos_range[1] - pos_range[0]) / (n_cells_per_dim * 0.5)
        
        # Generate centers directly as tensor
        self.centers = self._generate_centers().to(self.device)
        
        # Initialize decoder weights (will be trained later)
        self.weights = torch.nn.Parameter(torch.randn(self.total_cells, self.n_dims, device=self.device))
    
    @property
    @abstractmethod
    def n_dims(self):
        """Number of dimensions for this system."""
        pass
    
    @property
    def total_cells(self):
        """Total number of place cells."""
        return self.n_cells_per_dim ** self.n_dims
    
    @abstractmethod
    def _generate_centers(self):
        """Generate centers for place cells."""
        pass
    
    def compute_place_field(self, positions):
        """Compute the response of all place cells for given positions.
        
        Args:
            positions: tensor of shape (batch_size, n_dims) or (n_dims,)
        """
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
            
        positions = positions.to(self.device)
        
        # Compute distances for all positions at once
        # centers: (total_cells, n_dims)
        # positions: (batch_size, n_dims)
        # distances: (batch_size, total_cells)
        distances = torch.sqrt(torch.sum((self.centers.unsqueeze(0) - positions.unsqueeze(1)) ** 2, dim=2))
        # Compute Gaussian response
        responses = torch.exp(-0.5 * (distances / self.sigma) ** 2)
        
        return responses.squeeze()
    
    def decode_position(self, place_field):
        """Decode position from place cell responses."""
        if not isinstance(place_field, torch.Tensor):
            place_field = torch.tensor(place_field, dtype=torch.float32, device=self.device)
        if place_field.dim() == 1:
            place_field = place_field.unsqueeze(0)
            
        # Weighted sum for each dimension
        return torch.matmul(place_field, self.weights)
    
    @abstractmethod
    def plot_place_fields(self):
        """Plot the response curves of place cells."""
        pass
    
    @abstractmethod
    def plot_population_response(self, position):
        """Plot the population response for a specific position."""
        pass
    
    @abstractmethod
    def plot_decoder_weights(self):
        """Plot the learned decoder weights."""
        pass

class PlaceFieldSystem1D(PlaceFieldSystemBase):
    @property
    def n_dims(self):
        return 1
    
    def _generate_centers(self):
        """Generate 1D grid of place cell centers."""
        return torch.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim, dtype=torch.float32).reshape(-1, 1)
    
    def plot_place_fields(self):
        """Plot place fields for 1D case."""
        with torch.no_grad():
            # Generate positions on device
            positions = torch.linspace(self.pos_range[0], self.pos_range[1], 200, device=self.device).reshape(-1, 1)
            # Compute responses in one batch
            responses = self.compute_place_field(positions).cpu().numpy()
            positions = positions.squeeze().cpu().numpy()
            
            plt.figure(figsize=(10, 5))
            plt.plot(positions, responses)
            plt.title('1D Place Cell Response Curves')
            plt.xlabel('Position')
            plt.ylabel('Response')
            plt.grid(True)
            plt.show()
    
    def plot_decoder_weights(self):
        """Plot decoder weights for 1D case."""
        with torch.no_grad():
            weights = self.weights.cpu().numpy()
            centers = self.centers.cpu().flatten()
            
            plt.figure(figsize=(10, 4))
            plt.plot(centers, weights, 'b-', label='Weights')
            plt.scatter(centers, weights, c='b', alpha=0.6)
            plt.title('1D Decoder Weights')
            plt.xlabel('Cell Center Position')
            plt.ylabel('Weight')
            plt.grid(True)
            plt.legend()
            plt.show()
    
    def plot_population_response(self, position):
        """Plot population response for 1D case."""
        with torch.no_grad():
            if not isinstance(position, torch.Tensor):
                position = torch.tensor([position], dtype=torch.float32, device=self.device)
            responses = self.compute_place_field(position).cpu().numpy()
            
            plt.figure(figsize=(10, 4))
            plt.bar(np.arange(self.total_cells), responses, alpha=0.6)
            plt.axvline(x=(position.item() - self.pos_range[0]) / 
                       (self.pos_range[1] - self.pos_range[0]) * self.total_cells,
                       color='r', linestyle='--', label=f'True Position: {position.item():.2f}')
            plt.title(f'1D Population Response at Position {position.item():.2f}')
            plt.xlabel('Cell Index')
            plt.ylabel('Response')
            plt.grid(True)
            plt.legend()
            plt.show()

class PlaceFieldSystem2D(PlaceFieldSystemBase):
    @property
    def n_dims(self):
        return 2
    
    def _generate_centers(self):
        """Generate 2D grid of place cell centers."""
        x = torch.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim, dtype=torch.float32)
        y = torch.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    def plot_place_fields(self):
        """Plot example place fields for 2D case."""
        with torch.no_grad():
            x = torch.linspace(self.pos_range[0], self.pos_range[1], 50, device=self.device)
            y = torch.linspace(self.pos_range[0], self.pos_range[1], 50, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            
            # Compute responses for all positions at once
            all_responses = self.compute_place_field(positions)
            
            # Plot responses for a few example cells
            fig = plt.figure(figsize=(15, 5))
            example_cells = [0, self.total_cells // 4, self.total_cells // 2]
            
            for idx, cell_idx in enumerate(example_cells):
                ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
                responses = all_responses[:, cell_idx].reshape(50, 50).cpu().numpy()
                
                surf = ax.plot_surface(xx.cpu().numpy(), yy.cpu().numpy(), responses, cmap='viridis')
                plt.colorbar(surf, ax=ax)
                ax.set_title(f'2D Place Cell {cell_idx}')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_zlabel('Response')
            
            plt.tight_layout()
            plt.show()
    
    def plot_decoder_weights(self):
        """Plot decoder weights for 2D case."""
        with torch.no_grad():
            weights = self.weights.cpu().numpy()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Reshape weights for each output dimension
            weight_grid_x = weights[:, 0].reshape(self.n_cells_per_dim, self.n_cells_per_dim)
            weight_grid_y = weights[:, 1].reshape(self.n_cells_per_dim, self.n_cells_per_dim)
            
            # Plot X-dimension weights
            im1 = ax1.imshow(weight_grid_x, extent=[self.pos_range[0], self.pos_range[1],
                                                   self.pos_range[0], self.pos_range[1]],
                            origin='lower', cmap='RdBu_r')
            plt.colorbar(im1, ax=ax1, label='Weight Value')
            ax1.set_title('X-dimension Decoder Weights')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            
            # Plot Y-dimension weights
            im2 = ax2.imshow(weight_grid_y, extent=[self.pos_range[0], self.pos_range[1],
                                                   self.pos_range[0], self.pos_range[1]],
                            origin='lower', cmap='RdBu_r')
            plt.colorbar(im2, ax=ax2, label='Weight Value')
            ax2.set_title('Y-dimension Decoder Weights')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            
            plt.tight_layout()
            plt.show()
    
    def plot_population_response(self, position):
        """Plot population response for 2D case."""
        with torch.no_grad():
            if not isinstance(position, torch.Tensor):
                position = torch.tensor(position, dtype=torch.float32, device=self.device)
            responses = self.compute_place_field(position)
            response_grid = responses.reshape(self.n_cells_per_dim, self.n_cells_per_dim).cpu().numpy()
            pos_np = position.cpu().numpy()
            
            plt.figure(figsize=(8, 6))
            plt.imshow(response_grid, extent=[self.pos_range[0], self.pos_range[1],
                                          self.pos_range[0], self.pos_range[1]],
                    origin='lower', cmap='viridis')
            plt.plot(pos_np[0], pos_np[1], 'r*', markersize=15, 
                    label=f'True Position: ({pos_np[0]:.2f}, {pos_np[1]:.2f})')
            plt.colorbar(label='Response')
            plt.title('2D Population Response')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.show()

class PlaceFieldSystem3D(PlaceFieldSystemBase):
    @property
    def n_dims(self):
        return 3
    
    def _generate_centers(self):
        """Generate 3D grid of place cell centers."""
        x = torch.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim, dtype=torch.float32)
        y = torch.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim, dtype=torch.float32)
        z = torch.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim, dtype=torch.float32)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    
    def plot_place_fields(self):
        """Plot example place fields for 3D case using slices."""
        with torch.no_grad():
            x = torch.linspace(self.pos_range[0], self.pos_range[1], 20, device=self.device)
            y = torch.linspace(self.pos_range[0], self.pos_range[1], 20, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            # Plot responses for example cells at different z-slices
            fig = plt.figure(figsize=(15, 5))
            example_cell = self.total_cells // 2
            z_slices = [-0.5, 0.0, 0.5]
            
            # Create all positions at once for batch processing
            positions = []
            for z in z_slices:
                xy_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
                z_points = torch.full((xy_points.shape[0], 1), z, device=self.device)
                positions.append(torch.cat([xy_points, z_points], dim=1))
            
            all_positions = torch.cat(positions)
            all_responses = self.compute_place_field(all_positions)
            
            for idx, z in enumerate(z_slices):
                ax = fig.add_subplot(1, 3, idx + 1)
                start_idx = idx * 400  # 20x20=400 points per slice
                responses = all_responses[start_idx:start_idx + 400, example_cell].reshape(20, 20).cpu().numpy()
                
                im = ax.imshow(responses, extent=[self.pos_range[0], self.pos_range[1],
                                              self.pos_range[0], self.pos_range[1]],
                            origin='lower', cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'3D Place Cell at z={z:.1f}')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
            
            plt.tight_layout()
            plt.show()
    
    def plot_decoder_weights(self):
        """Plot decoder weights for 3D case."""
        with torch.no_grad():
            weights = self.weights.cpu().numpy()
            
            fig = plt.figure(figsize=(15, 4))
            titles = ['X-dimension', 'Y-dimension', 'Z-dimension']
            
            for dim in range(3):
                weight_cube = weights[:, dim].reshape(self.n_cells_per_dim, 
                                                    self.n_cells_per_dim,
                                                    self.n_cells_per_dim)
                
                # Show middle slice for each axis
                ax = fig.add_subplot(1, 3, dim + 1)
                slice_data = weight_cube[:, :, self.n_cells_per_dim//2]
                
                im = ax.imshow(slice_data, origin='lower', cmap='RdBu_r',
                              extent=[self.pos_range[0], self.pos_range[1],
                                     self.pos_range[0], self.pos_range[1]])
                plt.colorbar(im, ax=ax, label='Weight Value')
                ax.set_title(f'{titles[dim]} Weights\n(Middle Z-slice)')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
            
            plt.tight_layout()
            plt.show()
    
    def plot_population_response(self, position):
        """Plot population response for 3D case using orthogonal slices."""
        with torch.no_grad():
            if not isinstance(position, torch.Tensor):
                position = torch.tensor(position, dtype=torch.float32, device=self.device)
            responses = self.compute_place_field(position)
            response_cube = responses.reshape(self.n_cells_per_dim, 
                                           self.n_cells_per_dim,
                                           self.n_cells_per_dim).cpu().numpy()
            pos_np = position.cpu().numpy()
            
            # Plot three orthogonal slices through the true position
            fig = plt.figure(figsize=(15, 5))
            slice_names = ['YZ', 'XZ', 'XY']
            slices = [
                response_cube[self.n_cells_per_dim//2, :, :],
                response_cube[:, self.n_cells_per_dim//2, :],
                response_cube[:, :, self.n_cells_per_dim//2]
            ]
            
            for idx, (slice_data, name) in enumerate(zip(slices, slice_names)):
                ax = fig.add_subplot(1, 3, idx + 1)
                im = ax.imshow(slice_data, origin='lower', cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'3D Population Response ({name} plane)')
            
            plt.suptitle(f'Position: ({pos_np[0]:.2f}, {pos_np[1]:.2f}, {pos_np[2]:.2f})')
            plt.tight_layout()
            plt.show()

def plot_training_curves(losses, accuracies):
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    
    # Loss curve
    ax1.semilogy(losses)
    ax1.set_title('Training Loss (Log Scale)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss (log)')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.grid(True, which="major", ls="-", alpha=0.5)
    
    # Accuracy curve
    ax2.semilogy(accuracies)
    ax2.set_title('Decoding Error (Log Scale)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Euclidean Error (log)')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.grid(True, which="major", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def train_decoder(system, n_epochs=1000, batch_size=32, learning_rate=0.01, val_size=100, verbose=True):
    """Train the decoder weights using random positions."""
    optimizer = torch.optim.Adam([system.weights], lr=learning_rate)
    losses = []
    accuracies = []
    
    for epoch in range(n_epochs):
        # Generate random positions for training directly on device
        positions = torch.rand(batch_size, system.n_dims, device=system.device) * \
                   (system.pos_range[1] - system.pos_range[0]) + system.pos_range[0]
        
        # Compute place fields and decoded positions in one pass
        place_fields = system.compute_place_field(positions)
        decoded_positions = system.decode_position(place_fields)
        
        # Compute loss (MSE)
        loss = torch.mean(torch.sum((decoded_positions - positions) ** 2, dim=1))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Compute validation accuracy
        with torch.no_grad():
            # Generate validation positions directly on device
            val_positions = torch.rand(val_size, system.n_dims, device=system.device) * \
                          (system.pos_range[1] - system.pos_range[0]) + system.pos_range[0]
            
            # Compute validation error in one pass
            val_place_fields = system.compute_place_field(val_positions)
            val_decoded = system.decode_position(val_place_fields)
            errors = torch.sqrt(torch.sum((val_decoded - val_positions) ** 2, dim=1))
            mean_error = errors.mean().item()
            accuracies.append(mean_error)
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}, Mean Error: {mean_error:.4f}')
    
    return losses, accuracies

def evaluate_decoder(system, n_test=1000):
    """Evaluate decoder accuracy with test positions."""
    with torch.no_grad():
        # Generate test positions directly on device
        test_positions = torch.rand(n_test, system.n_dims, device=system.device) * \
                        (system.pos_range[1] - system.pos_range[0]) + system.pos_range[0]
        
        # Compute everything in one pass
        place_fields = system.compute_place_field(test_positions)
        decoded_positions = system.decode_position(place_fields)
        
        # Calculate errors
        errors = torch.sqrt(torch.sum((decoded_positions - test_positions) ** 2, dim=1))
        errors_np = errors.cpu().numpy()
        
        print(f"\nAccuracy Statistics ({system.n_dims}D):")
        print(f"Mean Error: {np.mean(errors_np):.4f}")
        print(f"Standard Deviation: {np.std(errors_np):.4f}")
        print(f"Median Error: {np.median(errors_np):.4f}")
        print(f"90th Percentile Error: {np.percentile(errors_np, 90):.4f}")
        
        return errors_np

def main_demo():
    # Test all three systems
    systems = [
        PlaceFieldSystem1D(n_cells_per_dim=50),
        PlaceFieldSystem2D(n_cells_per_dim=10),
        PlaceFieldSystem3D(n_cells_per_dim=7)
    ]
    
    for system in systems:
        print(f"\nTesting {system.n_dims}D system:")
        print(f"Total place cells: {system.total_cells}")
        
        # Show initial place fields
        print("\nPlace field visualization:")
        system.plot_place_fields()
        
        # Train the decoder
        print("\nTraining decoder...")
        losses, accuracies = train_decoder(system, verbose=True)
        plot_training_curves(losses, accuracies)
        
        # Show decoder weights
        print("\nDecoder weight visualization:")
        system.plot_decoder_weights()
        
        # Show example responses
        if system.n_dims == 1:
            positions = [-0.5, 0.0, 0.5]
        elif system.n_dims == 2:
            positions = [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]
        else:  # 3D
            positions = [[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        
        print("\nExample population responses:")
        for pos in positions:
            system.plot_population_response(pos)
        
        # Evaluate decoder
        evaluate_decoder(system)

def main_stat_1D(n_trials=100, n_cells=50):
    """
    Repeatedly test PlaceFieldSystem1D and analyze final mean errors.
    
    Args:
        n_trials: Number of trials/initializations
        n_cells: Number of cells per dimension
    """
    final_errors = []
    
    for trial in range(n_trials):
        # Create and train system
        system = PlaceFieldSystem1D(n_cells_per_dim=n_cells)
        losses, accuracies = train_decoder(system, verbose=False)
        final_errors.append(accuracies[-1])
    
    # Save histogram to disk
    plt.figure(figsize=(10, 6))
    plt.hist(final_errors, bins=30)
    plt.title('Distribution of Final Mean Errors')
    plt.xlabel('Mean Error')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('final_mean_errors_histogram.png')  # Save plot to disk
    
    # Display statistics
    print("\nError Statistics:")
    print(f"Mean: {np.mean(final_errors):.4f}")
    print(f"Std: {np.std(final_errors):.4f}")
    print(f"Min: {np.min(final_errors):.4f}")
    print(f"Max: {np.max(final_errors):.4f}")

if __name__ == "__main__":
    main_stat_1D()
