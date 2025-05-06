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
        # Ensure positions is at least 2D
        if positions.dim() == 0:  # scalar tensor
            positions = positions.view(1, 1)
        elif positions.dim() == 1:  # single position vector
            positions = positions.view(1, -1)
            
        # Compute squared distances directly using broadcasting
        squared_diff = (self.centers.unsqueeze(0) - positions.unsqueeze(1)) ** 2
        distances = torch.sqrt(squared_diff.sum(dim=-1))
        
        # Compute Gaussian response
        responses = torch.exp(-0.5 * (distances / self.sigma) ** 2)
        
        return responses.squeeze()
    
    def decode_position(self, place_field):
        """Decode position from place cell responses.
        
        Args:
            place_field: tensor of shape (batch_size, total_cells) or (total_cells,)
        """
        if place_field.dim() == 1:
            place_field = place_field.unsqueeze(0)
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
            # Generate positions and compute responses in single batch
            positions = torch.linspace(self.pos_range[0], self.pos_range[1], 200, 
                                     dtype=torch.float32, device=self.device).reshape(-1, 1)
            responses = self.compute_place_field(positions)
            
            # Transfer to CPU for plotting
            positions_np = positions.squeeze().cpu().numpy()
            responses_np = responses.cpu().numpy()
            
            plt.figure(figsize=(10, 5))
            plt.plot(positions_np, responses_np)
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
        """Plot population response for 1D case.
        
        Args:
            position: tensor of shape (1,) representing the position
        """
        with torch.no_grad():
            responses = self.compute_place_field(position).cpu().numpy()
            pos_value = position.item()
            
            plt.figure(figsize=(10, 4))
            plt.bar(np.arange(self.total_cells), responses, alpha=0.6)
            plt.axvline(x=(pos_value - self.pos_range[0]) / 
                       (self.pos_range[1] - self.pos_range[0]) * self.total_cells,
                       color='r', linestyle='--', label=f'True Position: {pos_value:.2f}')
            plt.title(f'1D Population Response at Position {pos_value:.2f}')
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
            # Generate grid positions with proper dtype
            x = torch.linspace(self.pos_range[0], self.pos_range[1], 50, 
                             dtype=torch.float32, device=self.device)
            y = torch.linspace(self.pos_range[0], self.pos_range[1], 50, 
                             dtype=torch.float32, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            
            # Compute responses for all positions at once
            all_responses = self.compute_place_field(positions)
            
            # Transfer to CPU once for plotting
            xx_np = xx.cpu().numpy()
            yy_np = yy.cpu().numpy()
            
            fig = plt.figure(figsize=(15, 5))
            example_cells = [0, self.total_cells // 4, self.total_cells // 2]
            
            for idx, cell_idx in enumerate(example_cells):
                ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
                responses = all_responses[:, cell_idx].reshape(50, 50).cpu().numpy()
                
                surf = ax.plot_surface(xx_np, yy_np, responses, cmap='viridis')
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
            # Reshape weights on GPU
            weights = self.weights.reshape(self.n_cells_per_dim, self.n_cells_per_dim, 2)
            # Transfer to CPU once
            weights_np = weights.cpu().numpy()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot X-dimension weights
            im1 = ax1.imshow(weights_np[..., 0], extent=[self.pos_range[0], self.pos_range[1],
                                                        self.pos_range[0], self.pos_range[1]],
                           origin='lower', cmap='RdBu_r')
            plt.colorbar(im1, ax=ax1, label='Weight Value')
            ax1.set_title('X-dimension Decoder Weights')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            
            # Plot Y-dimension weights
            im2 = ax2.imshow(weights_np[..., 1], extent=[self.pos_range[0], self.pos_range[1],
                                                        self.pos_range[0], self.pos_range[1]],
                           origin='lower', cmap='RdBu_r')
            plt.colorbar(im2, ax=ax2, label='Weight Value')
            ax2.set_title('Y-dimension Decoder Weights')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            
            plt.tight_layout()
            plt.show()
    
    def plot_population_response(self, position):
        """Plot population response for 2D case.
        
        Args:
            position: tensor of shape (2,) representing the 2D position
        """
        with torch.no_grad():
            responses = self.compute_place_field(position)
            response_grid = responses.reshape(self.n_cells_per_dim, self.n_cells_per_dim).cpu().numpy()
            pos_x, pos_y = position.cpu().numpy()
            
            plt.figure(figsize=(8, 6))
            plt.imshow(response_grid, extent=[self.pos_range[0], self.pos_range[1],
                                          self.pos_range[0], self.pos_range[1]],
                    origin='lower', cmap='viridis')
            plt.plot(pos_x, pos_y, 'r*', markersize=15, 
                    label=f'True Position: ({pos_x:.2f}, {pos_y:.2f})')
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
            # Generate grid positions with proper dtype
            x = torch.linspace(self.pos_range[0], self.pos_range[1], 20, 
                             dtype=torch.float32, device=self.device)
            y = torch.linspace(self.pos_range[0], self.pos_range[1], 20, 
                             dtype=torch.float32, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            # Plot responses for example cells at different z-slices
            fig = plt.figure(figsize=(15, 5))
            example_cell = self.total_cells // 2
            z_slices = torch.tensor([-0.5, 0.0, 0.5], device=self.device)
            points_per_slice = xx.numel()  # 20x20=400 points
            
            # Create all positions at once using broadcasting
            xy_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            z_points = z_slices.view(-1, 1, 1).expand(-1, points_per_slice, 1)
            positions = torch.cat([
                xy_points.unsqueeze(0).expand(len(z_slices), -1, -1),
                z_points
            ], dim=2)
            positions = positions.reshape(-1, 3)  # Flatten to (N, 3)
            
            # Compute all responses in one batch
            all_responses = self.compute_place_field(positions)
            
            # Transfer to CPU once for plotting
            xx_np = xx.cpu().numpy()
            yy_np = yy.cpu().numpy()
            responses_np = all_responses[:, example_cell].cpu().numpy()
            
            for idx, z in enumerate(z_slices.cpu().numpy()):
                ax = fig.add_subplot(1, 3, idx + 1)
                start_idx = idx * points_per_slice
                slice_responses = responses_np[start_idx:start_idx + points_per_slice].reshape(20, 20)
                
                im = ax.imshow(slice_responses, extent=[self.pos_range[0], self.pos_range[1],
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
            # Reshape weights on GPU
            weights = self.weights.reshape(self.n_cells_per_dim, 
                                        self.n_cells_per_dim,
                                        self.n_cells_per_dim, 3)
            # Extract middle slices for each dimension
            middle_slices = weights[:, :, self.n_cells_per_dim//2, :].cpu().numpy()
            
            fig = plt.figure(figsize=(15, 4))
            titles = ['X-dimension', 'Y-dimension', 'Z-dimension']
            
            for dim in range(3):
                ax = fig.add_subplot(1, 3, dim + 1)
                im = ax.imshow(middle_slices[..., dim], origin='lower', cmap='RdBu_r',
                             extent=[self.pos_range[0], self.pos_range[1],
                                    self.pos_range[0], self.pos_range[1]])
                plt.colorbar(im, ax=ax, label='Weight Value')
                ax.set_title(f'{titles[dim]} Weights\n(Middle Z-slice)')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
            
            plt.tight_layout()
            plt.show()
    
    def plot_population_response(self, position):
        """Plot population response for 3D case using orthogonal slices.
        
        Args:
            position: tensor of shape (3,) representing the 3D position
        """
        with torch.no_grad():
            responses = self.compute_place_field(position)
            response_cube = responses.reshape(self.n_cells_per_dim, 
                                           self.n_cells_per_dim,
                                           self.n_cells_per_dim).cpu().numpy()
            pos_x, pos_y, pos_z = position.cpu().numpy()
            
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
            
            plt.suptitle(f'Position: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})')
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

def train_decoder(system, n_epochs=100, batch_size=320, learning_rate=0.1, val_size=100, verbose=True):
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
        
        # Show example responses with tensor positions
        print("\nExample population responses:")
        if system.n_dims == 1:
            positions = torch.tensor([-0.5, 0.0, 0.5], device=system.device)
            for pos in positions:
                system.plot_population_response(pos)
        elif system.n_dims == 2:
            positions = torch.tensor([
                [-0.5, -0.5],
                [0.0, 0.0],
                [0.5, 0.5]
            ], device=system.device)
            for pos in positions:
                system.plot_population_response(pos)
        else:  # 3D
            positions = torch.tensor([
                [-0.5, -0.5, -0.5],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5]
            ], device=system.device)
            for pos in positions:
                system.plot_population_response(pos)
        
        # Evaluate decoder
        evaluate_decoder(system)

def main_stat_1D(n_trials=100, n_cells=50, show_plot=False):
    """
    Repeatedly test PlaceFieldSystem1D and analyze final mean errors.
    
    Args:
        n_trials: Number of trials/initializations
        n_cells: Number of cells per dimension
        show_plot: Whether to show plot (True) or save to disk (False)
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
    if show_plot:
        plt.show()
    else:
        plt.savefig('final_mean_errors_histogram.png')  # Save plot to disk
    
    # Display statistics
    print("\nError Statistics:")
    print(f"Mean: {np.mean(final_errors):.4f}")
    print(f"Std: {np.std(final_errors):.4f}")
    print(f"Min: {np.min(final_errors):.4f}")
    print(f"Max: {np.max(final_errors):.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Place field system analysis and demonstration')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parser for stat command
    stat_parser = subparsers.add_parser('stat', help='Run statistical analysis on 1D system')
    stat_parser.add_argument('--show', action='store_true', help='Show plot instead of saving to disk')
    
    # Parser for demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration of all systems')
    
    args = parser.parse_args()
    
    if args.command == 'stat':
        main_stat_1D(show_plot=args.show)
    elif args.command == 'demo':
        main_demo()
    else:
        parser.print_help()
